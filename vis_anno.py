import argparse
import cv2
import neural_renderer as nr
import os
import random
import numpy as np 
import shutil
import torch
import trimesh
import json
import mano
import matplotlib.pyplot as plt
from mow_utils import PerspectiveRenderer, center_vertices, projection

from utils.vis_utils import render_mesh

render_type = 'both' # ['hand', 'object', 'both']


with open('poses.json', 'r') as f:
    annos = json.load(f)

for anno in annos:
    image_id = anno['image_id']
    obj_path = 'data/models/{}.obj'.format(image_id)
    im_path = 'data/images/{}.jpg'.format(image_id)

    print("Loading image...")
    im = cv2.imread(im_path)[:, :, ::-1]
    im_size = max(im.shape[:2])
    renderer = PerspectiveRenderer(im_size)
    renderer.set_light_dir([1, 0.5, 1], 0.3, 0.5)
    renderer.set_bgcolor([1, 1, 1])


    print("Loading object...")
    obj_mesh = trimesh.load(obj_path, force='mesh')
    obj_verts, obj_faces = torch.from_numpy(obj_mesh.vertices).cuda().float(), torch.from_numpy(obj_mesh.faces).cuda().float()

    # Normalize
    obj_verts -= obj_verts.min(0)[0][None, :]
    obj_verts /= torch.abs(obj_verts).max()
    obj_verts *= 2
    obj_verts -= obj_verts.max(0)[0][None, :] / 2

    # Center mesh
    verts, faces = center_vertices(obj_verts, obj_faces)

    # Render object
    mesh_obj = PerspectiveRenderer.Model(
        renderer,
        verts,
        faces,
        translation=torch.tensor(anno['t']).reshape((1, 3)).to('cuda:0'),
        rotation=torch.tensor(anno['R']).reshape((3, 3)).to('cuda:0'),
        scale=anno['s'],
        color_name='red'
    )

    # render_mesh(im, mesh_obj.vertices.detach().cpu().numpy()[0], mesh_obj.faces.detach().cpu().numpy()[0], np.eye(4))
    default_K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
    dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]])
    orig_size = 1

    print("Loading hand...")
    rhm_path = 'mano/MANO_RIGHT.pkl'
    rh_model = mano.load(
        model_path=rhm_path,
        model_type='mano',
        num_pca_comps=45,
        batch_size=1,
        flat_hand_mean=False
    )

    mano_pose = torch.tensor(anno['hand_pose']).unsqueeze(0)
    transl = torch.tensor(anno['trans']).unsqueeze(0)
    pose, global_orient = mano_pose[:, 3:], mano_pose[:, :3]

    mesh_hand = rh_model(
        global_orient=global_orient,
        hand_pose=pose,
        transl=transl,
        return_verts=True,
        return_tips=True
    )

    hand = PerspectiveRenderer.Model(
        renderer,
        mesh_hand.vertices.detach().squeeze(0).to('cuda:0'),
        torch.tensor(rh_model.faces.astype('int32')).to('cuda:0'),
        translation=torch.tensor(anno['hand_t']).reshape((1, 3)).to('cuda:0'),
        rotation=torch.tensor(anno['hand_R']).reshape((3, 3)).type(torch.float32).to('cuda:0'),
        scale=anno['hand_s'],
        color_name='blue'
    )

    print("Rendering...")
    if render_type == 'hand':
        im_vis, sil_vis = renderer(
            [hand],
            image=im
        )
    elif render_type == 'object':
        im_vis, sil_vis = renderer(
            [mesh_obj],
            image=im
        )
    elif render_type == 'both':
        im_vis, sil_vis = renderer(
            [mesh_obj, hand],
            image=im
        )
    else:
        raise NotImplementedError

    cv2.imwrite(os.path.join('data', 'renders', render_type, f'{image_id}.jpg'), im_vis[..., ::-1]*255.)
    cv2.imwrite(os.path.join('data', 'masks', render_type, f'{image_id}.jpg'), (sil_vis > 0.5)*255.)