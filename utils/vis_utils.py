import cv2
import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt


def render_mesh(img, mesh, face, cam_param, cam_pose=None):
    # mesh
    orig_mesh = mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
	np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')
    
    focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    if cam_pose is not None:
        scene.add(camera, pose=cam_pose)
    else:
        scene.add(camera)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)
   
    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    # save to image
    img = rgb * valid_mask + img * (1-valid_mask)
    return img


def vis_keypoints(img, kps, alpha=1, size=3):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    if len(kps) == 1: colors = [(255,255,255)]

    if kps.shape[1] == 2:
        kps = np.concatenate([kps, np.ones((len(kps),1))], axis=1)

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    kp_mask = np.ascontiguousarray(kp_mask, dtype=np.uint8)

    # Draw the keypoints.
    for i in range(len(kps)):
        if kps[i][-1] > 0:
            p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
            cv2.circle(kp_mask, p, radius=size, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
            
    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)