import os
import pycolmap
import numpy as np

# Path to the directory containing your images
image_dir = '/root/dataset/NeRF/nerf_synthetic/bed'

# Initialize reconstruction object
reconstruction = pycolmap.Reconstruction()

# Add images to reconstruction
images = os.listdir(image_dir)
for i, image_path in enumerate(images):
    im_path = os.path.join(image_dir, image_path)
    image = pycolmap.Image(
        id=i,
        name=im_path,
    )
    reconstruction.add_image(image)

# Extract features and matches
extractor = pycolmap.FeatureExtractor(reconstruction)
extractor.extract_features_and_matches()

# Reconstruct the scene
mapper = pycolmap.Mapper(reconstruction)
mapper.triangulate()
mapper.bundle_adjust()

# Save the reconstructed scene to file
reconstruction_path = 'reconstruction.bin'
pycolmap.write_model(reconstruction_path, reconstruction)

# Load the reconstructed scene from file
reconstruction = pycolmap.read_model(reconstruction_path)

# Get camera poses
camera_ids = reconstruction.camera_ids
camera_poses = []
for camera_id in camera_ids:
    camera = reconstruction.cameras[camera_id]
    camera_pose = np.linalg.inv(camera.model_to_world)
    camera_poses.append(camera_pose)

print(camera_poses)
