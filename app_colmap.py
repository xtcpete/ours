import argparse
import torch
from src.loftr import LoFTR
from src.config.default import get_cfg_defaults
from src.config.default_aspan import get_cfg_defaults as get_cfg_defaults_aspan
from src.ASpanFormer.aspanformer import ASpanFormer
from src.utils.misc import lower_config, move_to_device
from src.utils.dataset import read_megadepth_color, read_megadepth_gray, load_color_image, getImageSize
from src.utils.plot_gt import make_matching_figure
import matplotlib.cm as cm
import os
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # data_path for training data
    parser.add_argument("data_path", type=str, default="test", help="data path")

    # weight path
    parser.add_argument("--load_weights", type=str, default='weights/outdoor_loftr.ckpt', help="load weights")

    # model 
    parser.add_argument("--model", type=str, default='loftr', help="model name")

    parser.add_argument("--dump_path", type=str, default='colmap_results', help="save path")

    parser.add_argument("--visualization", action='store_true', default=False, help="visualization")

    return parser.parse_args()

def main():
    args = parse_args()
    print(args)
    weight_path = args.load_weights
    data_dir = args.data_path
    if args.model == 'loftr':
        config = lower_config(get_cfg_defaults())
        matcher = LoFTR(lower_config(config['loftr']))
        imread = read_megadepth_color
    elif args.model == 'aspan':
        config = lower_config(get_cfg_defaults_aspan())
        matcher = ASpanFormer(lower_config(config['aspan']))
        imread = read_megadepth_gray
        assert 'aspan' in weight_path, 'Please use the aspanformer weights!'
    print("Model initialized successfully!")
    weights = torch.load(weight_path, map_location="cpu")["state_dict"]
    # remove the first 'module.' in the state_dict
    weights = {k.replace('matcher.', ''): v for k, v in weights.items()}
    matcher.load_state_dict(weights)
    matcher.to('cuda')
    matcher.eval()
    print("Load weights successfully!")

    if not os.path.exists(args.dump_path):
        os.mkdir(args.dump_path)
        os.mkdir(os.path.join(args.dump_path, 'images'))    
        os.mkdir(os.path.join(args.dump_path, 'visualization'))
    elif not os.path.exists(os.path.join(args.dump_path, 'images')):
        os.mkdir(os.path.join(args.dump_path, 'images'))
        os.mkdir(os.path.join(args.dump_path, 'visualization'))

    images = os.listdir(data_dir)
    images_points = dict()
    pairs = dict()
    for i in range(len(images)):
        image0_path = os.path.join(data_dir, images[i])
        # image path
        image0, mask0, scale0, oshape0 = imread(image0_path, 900, 8, True)
        image0 = image0[None]
        scale0 = scale0.unsqueeze(0)
        for j in range(i+1, len(images)):
            print('Evaluate the pair: ', images[i], images[j], '...')
            image1_path = os.path.join(data_dir, images[j])
            image1, mask1, scale1, oshape1 = imread(image1_path, 900, 8, True)
            image1 = image1[None]
            scale1 = scale1.unsqueeze(0)
            batch = ({'image0': image0.to('cuda'), 'image1': image1.to('cuda'), 'scale0': scale0, 'scale1': scale1})
            move_to_device(batch, 'cuda')
            with torch.no_grad():
                matcher(batch)
                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()

                image0_color, _ = load_color_image(image0_path)
                image1_color, _ = load_color_image(image1_path)
                index = np.where(mconf >= 0.2)[0]
                fname0_hw = getImageSize(image0_path)
                fname1_hw = getImageSize(image1_path)
                discard_index0 = np.where(mkpts0[:, 1] >= (fname0_hw[0]-5))[0]
                discard_index1 = np.where(mkpts0[:, 0] >= (fname0_hw[1]-5))[0]
                discard_index2 = np.where(mkpts1[:, 1] >= (fname1_hw[0]-5))[0]
                discard_index3 = np.where(mkpts1[:, 0] >= (fname1_hw[1]-5))[0]

                index = np.setdiff1d(index, discard_index0)
                index = np.setdiff1d(index, discard_index1)
                index = np.setdiff1d(index, discard_index2)
                index = np.setdiff1d(index, discard_index3)

                color = cm.jet_r(mconf, alpha=0.7)
                text = [
                    'Ours',
                    'Matches: {}'.format(len(index))
                ]
                if len(index) > 10:
                    print(f'Found {len(index)} matches between {images[i]} and {images[j]}')
                    name = f'{images[i]}AND{images[j]}'
                    pairs[name] = {'i_ids': batch['i_ids'][index].cpu().numpy().tolist(), 'j_ids': batch['j_ids'][index].cpu().numpy().tolist(), 'mkpts0': mkpts0[index].tolist(), 'mkpts1': mkpts1[index].tolist(), 'mconf': mconf[index].tolist()}
                    if args.visualization:
                        svae_path = os.path.join(f'{args.dump_path}/visualization/{images[i]}AND{images[j]}.png')
                        make_matching_figure(image0_color, image1_color, mkpts0[index], mkpts1[index], color = color[index], text = text, dpi=100, path=svae_path)  

    # getting all points
    images_points = {}
    print('Writing text files...')
    for name in pairs.keys():
        keys = name.split('AND')
        image0 = keys[0]
        image1 = keys[1]
        if image0 not in images_points.keys():
            images_points[image0] = np.array(pairs[name]['mkpts0'])
        else:
            images_points[image0] = np.unique(np.concatenate((images_points[image0], np.array(pairs[name]['mkpts0'])), axis=0), axis=0)
        
        if image1 not in images_points.keys():
            images_points[image1] = np.array(pairs[name]['mkpts1'])
        else:
            images_points[image1] = np.unique(np.concatenate((images_points[image1], np.array(pairs[name]['mkpts1'])), axis=0), axis=0)


    with open(f'{args.dump_path}/matching.txt', 'w') as f:
        for key in pairs.keys():
            names = key.split('AND')
            image0 = names[0]
            image1 = names[1]
            
            mconf = pairs[key]['mconf']
            index = np.where(np.array(mconf) > 0.2)[0]
            f.write(image0 + ' ' + image1 + '\n')

            mkpts0 = np.array(pairs[key]['mkpts0'])[index]
            mkpts1 = np.array(pairs[key]['mkpts1'])[index]
            
            for i in range(mkpts0.shape[0]):
                x0 = np.where(np.all(images_points[image0] == mkpts0[i], axis=1))[0][0]
                x1 = np.where(np.all(images_points[image1] == mkpts1[i], axis=1))[0][0]
                f.write(str(x0) + ' ' + str(x1) + '\n')
            f.write('\n')   
    print('Matching file written!')


    for key in images_points.keys():
        image_points = images_points[key]
        with open(f'{args.dump_path}/images/{key}.txt', 'w') as f:
            line = str(image_points.shape[0]) + ' 128\n'
            f.write(line)
            for i in range(image_points.shape[0]):
                x = image_points[i][0]
                y = image_points[i][1]

                line = ["{:.2f}".format(x), "{:.2f}".format(y), '1', '0'] + ['0'] * 128
                f.write(' '.join(line) + '\n')
    print('Done!')
if __name__ == '__main__':
    main()