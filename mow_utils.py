#!/usr/bin/env python3

"""Utility functions."""
from __future__ import division

import cv2
import math
import neural_renderer as nr
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy


def compute_R_init(B):
    """Computes B random rotation matrices (source: Zhang et al. 2020)."""
    x1, x2, x3 = torch.split(torch.rand(3 * B).cuda(), B)
    tau = 2 * math.pi
    R = torch.stack(
        (  # B x 3 x 3
            torch.stack(
                (torch.cos(tau * x1), torch.sin(tau * x1), torch.zeros_like(x1)), 1
            ),
            torch.stack(
                (-torch.sin(tau * x1), torch.cos(tau * x1), torch.zeros_like(x1)), 1
            ),
            torch.stack(
                (torch.zeros_like(x1), torch.zeros_like(x1), torch.ones_like(x1)), 1
            ),
        ),
        1,
    )
    v = torch.stack(
        (  # B x 3
            torch.cos(tau * x2) * torch.sqrt(x3),
            torch.sin(tau * x2) * torch.sqrt(x3),
            torch.sqrt(1 - x3),
        ),
        1,
    )
    identity = torch.eye(3).repeat(B, 1, 1).cuda()
    H = identity - 2 * v.unsqueeze(2) * v.unsqueeze(1)
    rotation_matrices = -torch.matmul(H, R)
    return rotation_matrices


def compute_bbox_proj(verts, f, img_size=256):
    """Computes bbox of projected verts (source: Zhang et al. 2020)."""
    xy = verts[:, :, :2]
    z = verts[:, :, 2:]
    proj = f * xy / z + 0.5  # [0, 1]
    proj = proj * img_size  # [0, img_size]
    u, v = proj[:, :, 0], proj[:, :, 1]
    x1, x2 = u.min(1).values, u.max(1).values
    y1, y2 = v.min(1).values, v.max(1).values
    return torch.stack((x1, y1, x2 - x1, y2 - y1), 1)


def compute_t_init(bbox_target, vertices, f=1, img_size=256):
    """Computes initial translation (source: Zhang et al. 2020)."""
    bbox_mask = np.array(bbox_target)
    mask_center = bbox_mask[:2] + bbox_mask[2:] / 2
    diag_mask = np.sqrt(bbox_mask[2] ** 2 + bbox_mask[3] ** 2)
    B = vertices.shape[0]
    x = torch.zeros(B).cuda()
    y = torch.zeros(B).cuda()
    z = 2.5 * torch.ones(B).cuda()
    for _ in range(50):
        translation = torch.stack((x, y, z), -1).unsqueeze(1)
        v = vertices + translation
        bbox_proj = compute_bbox_proj(v, f=1, img_size=img_size)
        diag_proj = torch.sqrt(torch.sum(bbox_proj[:, 2:] ** 2, 1))
        delta_z = z * (diag_proj / diag_mask - 1)
        z = z + delta_z
        proj_center = bbox_proj[:, :2] + bbox_proj[:, 2:] / 2
        x += (mask_center[0] - proj_center[:, 0]) * z / f / img_size
        y += (mask_center[1] - proj_center[:, 1]) * z / f / img_size
    return torch.stack((x, y, z), -1).unsqueeze(1)


def matrix_to_rot6d(rotmat):
    """Converts rot 3x3 mat to 6D (source: Zhang et al. 2020)."""
    return rotmat.view(-1, 3, 3)[:, :, :2]


def rot6d_to_matrix(rot_6d):
    """Converts 6D rot to 3x3 mat (source: Zhang et al. 2020)."""
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)



def center_vertices(vertices, faces, flip_y=True):
    """Centroid-align vertices."""
    vertices = vertices - vertices.mean(dim=0, keepdim=True)
    if flip_y:
        vertices[:, 1] *= -1
        faces = faces[:, [2, 1, 0]]
    return vertices, faces




class PerspectiveRenderer(object):
    """Perspective renderer (source: Zhang et al. 2020)."""

    colors = {
        "blue": [0.65098039, 0.74117647, 0.85882353],
        "red": [251 / 255.0, 128 / 255.0, 114 / 255.0],
    }

    class Model(object):
        def __init__(
            self,
            renderer,
            vertices,
            faces,
            textures=None,
            translation=None,
            rotation=None,
            scale=None,
            color_name="white",
        ):
            if vertices.ndimension() == 2:
                vertices = vertices.unsqueeze(0)
            if faces.ndimension() == 2:
                faces = faces.unsqueeze(0)
            if textures is None:
                textures = torch.ones(
                    len(faces),
                    faces.shape[1],
                    renderer.t_size,
                    renderer.t_size,
                    renderer.t_size,
                    3,
                    dtype=torch.float32,
                ).cuda()
                color = torch.FloatTensor(renderer.colors[color_name]).cuda()
                textures = color * textures
            elif textures.ndimension() == 5:
                textures = textures.unsqueeze(0)

            if translation is None:
                translation = renderer.default_translation
            if not isinstance(translation, torch.Tensor):
                translation = torch.FloatTensor(translation).to(vertices.device)
            if translation.ndimension() == 1:
                translation = translation.unsqueeze(0)

            if rotation is not None:
                vertices = torch.matmul(vertices, rotation)
            vertices += translation

            if scale is not None:
                vertices *= scale
            
            self.vertices = vertices
            self.faces = faces
            self.textures = textures
        
        def join(self, model):
            self.faces = torch.cat((self.faces, model.faces + self.vertices.shape[1]), 1)
            self.vertices = torch.cat((self.vertices, model.vertices), 1)
            self.textures = torch.cat((self.textures, model.textures), 1)

    def __init__(self, image_size=256, texture_size=1):
        self.image_size = image_size
        self.default_K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
        self.R = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        self.t = torch.zeros(1, 3).cuda()
        self.default_translation = torch.cuda.FloatTensor([[0, 0, 2]])
        self.t_size = texture_size
        self.renderer = Renderer(
            image_size=image_size, K=self.default_K, R=self.R, t=self.t, orig_size=1 # cam mode: projection
        )
        self.set_light_dir([1, 1, 0.4], int_dir=0.3, int_amb=0.7)
        self.set_bgcolor([0, 0, 0])

    def __call__(
        self,
        models,
        image=None,
        K=None,
    ):
        if K is not None:
            self.renderer.K = K

        all_models = None
        for model in models:
            if all_models is None:
                all_models = model
            else:
                all_models.join(model)

        rend, depth, sil = self.renderer.render(all_models.vertices, all_models.faces, all_models.textures)
        rend = rend.detach().cpu().numpy().transpose(0, 2, 3, 1)  # B x H x W x C
        rend = np.clip(rend, 0, 1)[0]

        self.renderer.K = self.default_K  # Restore just in case.
        
        # Render on image
        sil = sil.detach().cpu().numpy()[0]
        h, w, *_ = image.shape
        L = max(h, w)
        if image.max() > 1:
            image = image.astype(float) / 255.0


        new_image = np.pad(image, ((0, L - h), (0, L - w), (0, 0)))
        new_image = cv2.resize(new_image, (self.image_size, self.image_size))
        new_sil = cv2.resize(sil, (self.image_size, self.image_size))
        new_image[sil > 0] = rend[sil > 0]
        r = self.image_size / L
        assert new_image.shape[:2] == new_sil.shape[:2]
        new_image = new_image[: int(h * r), : int(w * r)]
        new_sil = new_sil[: int(h * r), : int(w * r)]
        return new_image, new_sil

    def set_light_dir(self, direction=(1, 0.5, -1), int_dir=0.3, int_amb=0.7):
        self.renderer.light_direction = direction
        self.renderer.light_intensity_directional = int_dir
        self.renderer.light_intensity_ambient = int_amb

    def set_bgcolor(self, color):
        self.renderer.background_color = color


def vis_obj_pose_im(verts, faces, rot, t, scale, im, out_f, idx=0):
    im_size = max(im.shape[:2])
    renderer = PerspectiveRenderer(im_size)
    renderer.set_light_dir([1, 0.5, 1], 0.3, 0.5)
    renderer.set_bgcolor([1, 1, 1])
    im_vis = renderer(
        vertices=verts,
        faces=faces,
        image=im,
        translation=t[idx],
        rotation=rot[idx],
        scale=scale,
        color_name="red"
    )
    im_vis = im_vis[:, :, ::-1] * 255
    cv2.imwrite(out_f, im_vis)
    print("Wrote vis obj pose im to: {}".format(out_f))
    return im_vis





def projection(vertices, K, R, t, dist_coeffs, orig_size, eps=1e-9):
    '''
    Calculate projective transformation of vertices given a projection matrix
    Input parameters:
    K: batch_size * 3 * 3 intrinsic camera matrix
    R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
    dist_coeffs: vector of distortion coefficients
    orig_size: original size of image captured by the camera
    Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
    pixels and z is the depth
    '''

    # instead of P*x we compute x'*P'
    vertices = torch.matmul(vertices, R.transpose(2,1)) + t
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + eps)
    y_ = y / (z + eps)

    # Get distortion coefficients from vector
    k1 = dist_coeffs[:, None, 0]
    k2 = dist_coeffs[:, None, 1]
    p1 = dist_coeffs[:, None, 2]
    p2 = dist_coeffs[:, None, 3]
    k3 = dist_coeffs[:, None, 4]

    # we use x_ for x' and x__ for x'' etc.
    r = torch.sqrt(x_ ** 2 + y_ ** 2)
    x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
    y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
    vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
    vertices = torch.matmul(vertices, K.transpose(1,2))
    u, v = vertices[:, :, 0], vertices[:, :, 1]
    v = orig_size - v
    # map u,v from [0, img_size] to [-1, 1] to use by the renderer
    u = 2 * (u - orig_size / 2.) / orig_size
    v = 2 * (v - orig_size / 2.) / orig_size
    vertices = torch.stack([u, v, z], dim=-1)
    return vertices



def vertices_to_faces(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3)
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]




class Renderer(nn.Module):
    def __init__(self, image_size=256, anti_aliasing=True, background_color=[0,0,0],
                 fill_back=True, camera_mode='projection',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=1024,
                 perspective=True, viewing_angle=30, camera_direction=[0,0,1],
                 near=0.1, far=100,
                 light_intensity_ambient=0.5, light_intensity_directional=0.5,
                 light_color_ambient=[1,1,1], light_color_directional=[1,1,1],
                 light_direction=[0,1,0]):
        super(Renderer, self).__init__()
        # rendering
        self.image_size = image_size
        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back

        # camera
        self.camera_mode = camera_mode
        
        # projection
        self.K = K
        self.R = R
        self.t = t
        if isinstance(self.K, numpy.ndarray):
            self.K = torch.cuda.FloatTensor(self.K)
        if isinstance(self.R, numpy.ndarray):
            self.R = torch.cuda.FloatTensor(self.R)
        if isinstance(self.t, numpy.ndarray):
            self.t = torch.cuda.FloatTensor(self.t)
        self.dist_coeffs = dist_coeffs
        if dist_coeffs is None:
            self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]])
        self.orig_size = orig_size


        self.near = near
        self.far = far

        # light
        self.light_intensity_ambient = light_intensity_ambient
        self.light_intensity_directional = light_intensity_directional
        self.light_color_ambient = light_color_ambient
        self.light_color_directional = light_color_directional
        self.light_direction = light_direction 

        # rasterization
        self.rasterizer_eps = 1e-3


    def render(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # viewpoint transformation
        if K is None:
            K = self.K
        if R is None:
            R = self.R
        if t is None:
            t = self.t
        if dist_coeffs is None:
            dist_coeffs = self.dist_coeffs
        if orig_size is None:
            orig_size = self.orig_size
        vertices = projection(vertices, K, R, t, dist_coeffs, orig_size)

        # from utils.vis_utils import vis_keypoints
        # image = np.zeros((720, 1280, 3))
        # vertices = vertices[0, :, :2]
        # vertices[:, 0] = vertices[:, 0] * 1280
        # vertices[:, 1] = vertices[:, 1] * 720
        
        # vis_image= vis_keypoints(image, vertices.detach().cpu().numpy())
        # cv2.imwrite('vis_image.png', vis_image)
        # import pdb; pdb.set_trace()

        # rasterization
        faces = vertices_to_faces(vertices, faces)
        out = nr.rasterize_rgbad(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return out['rgb'], out['depth'], out['alpha']