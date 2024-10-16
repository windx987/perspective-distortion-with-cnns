import gradio as gr
import torch
import cv2
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn

def find_homography(src_pts, dst_pts):
  """
  Calculates the homography matrix between two sets of corresponding points.

  Args:
    src_pts: Source points (Nx2 array).
    dst_pts: Destination points (Nx2 array).

  Returns:
    The homography matrix (3x3 array).
  """

  assert src_pts.shape == dst_pts.shape and src_pts.shape[0] >= 4

  A = []
  for i in range(src_pts.shape[0]):
    x, y = src_pts[i]
    u, v = dst_pts[i]
    A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])

  A = np.array(A)
  _, _, V = np.linalg.svd(A)
  H = V[-1, :].reshape((3, 3))
  H = H / H[2, 2] 

  return H


def apply_homography(image, H):
  """
  Applies a homography transformation to an image.

  Args:
    image: The input image.
    H: The homography matrix.

  Returns:
    The transformed image.
  """

  rows, cols = image.shape[:2]
  new_corners = np.array([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])
  transformed_corners = cv2.perspectiveTransform(np.array([new_corners]).astype(np.float32), H)
  min_x = np.min(transformed_corners[:, :, 0])
  max_x = np.max(transformed_corners[:, :, 0])
  min_y = np.min(transformed_corners[:, :, 1])
  max_y = np.max(transformed_corners[:, :, 1])

  transformed_image = cv2.warpPerspective(image, H, (int(max_x - min_x), int(max_y - min_y)))
  return transformed_image

def warpPerspective(img, H, output_shape):
    """
    Applies a perspective transformation to an image using a homography matrix with optimized performance.

    Args:
        img: The input image (numpy array).
        H: The 3x3 homography matrix.
        output_shape: The desired shape of the output image (height, width).

    Returns:
        The warped image (numpy array).
    """
    height, width = output_shape

    # Create a grid of (x, y) coordinates in the output image
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Stack and reshape to form homogeneous coordinates (x, y, 1)
    homogeneous_coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones_like(x_coords).ravel()], axis=1)

    # Apply the inverse homography matrix to the coordinates
    src_coords = np.dot(homogeneous_coords, np.linalg.inv(H).T)

    # Normalize homogeneous coordinates to get (x, y)
    src_coords /= src_coords[:, 2][:, np.newaxis]

    # Clip the coordinates to ensure they are within image bounds
    src_x = np.clip(src_coords[:, 0].astype(np.int32), 0, img.shape[1] - 1)
    src_y = np.clip(src_coords[:, 1].astype(np.int32), 0, img.shape[0] - 1)

    # Reshape back to the output image shape
    warped_img = img[src_y, src_x].reshape(height, width, img.shape[2])

    return warped_img

class PointEstimatorCNN(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), features=[64, 128, 256, 384, 512, 1024, 2048], dims=[2048], final_dim=8):
        super(PointEstimatorCNN, self).__init__()

        layers = []
        input_channels = input_shape[0]
        for output_channels in features:
            layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(output_channels, affine=True))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            input_channels = output_channels
        
        layers.append(nn.Conv2d(input_channels, 2048, kernel_size=1, padding=1))
        self.features = nn.Sequential(*layers)
        
        self.fc = nn.Sequential(*([nn.Linear(self.get_flatten_dim(input_shape, self.features), final_dim)]))

    def get_flatten_dim(self, input_shape, features):
        x = torch.zeros(1, *input_shape)
        # print(x.shape)
        x = features(x)
        # print(x.shape)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = PointEstimatorCNN()
model.load_state_dict(torch.load('/home/teerawat.c/homography-projects/cp/0.2993_epoch32.pth'))
model.eval()  # Set the model to evaluation mode

# Preprocessing the image for the model (adjust based on your model's input size)
def preprocess_image(image):
    # Resize or transform the image as per the model's requirement
    image = cv2.resize(image, (224, 224))  # Example resize to 128x128, adjust based on your input
    # Convert to tensor, normalize, etc.
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_homography(image):
    # Preprocess the image
    input_tensor = preprocess_image(image)
    
    # Move the image to GPU if available
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        model.cuda()
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        
    # Move the tensor to the CPU before converting to NumPy
    points = output.squeeze().detach().cpu().numpy()
    
    A4_WIDTH_MM, A4_HEIGHT_MM, dpi = 297, 210, 96
    A4_WIDTH_PIXELS = int((A4_WIDTH_MM / 25.4) * dpi)
    A4_HEIGHT_PIXELS = int((A4_HEIGHT_MM / 25.4) * dpi)

    # Convert the points to a NumPy array
    src_pts = np.array(points, dtype=np.float32)
    src_pts = src_pts.reshape(4, 2)

    # Destination points: A4 corners in pixel space (top-left, top-right, bottom-right, bottom-left)
    dst_pts = np.array([
        [0, 0],  # Top-left corner
        [A4_WIDTH_PIXELS - 1, 0],  # Top-right corner
        [A4_WIDTH_PIXELS - 1, A4_HEIGHT_PIXELS - 1],  # Bottom-right corner
        [0, A4_HEIGHT_PIXELS - 1]  # Bottom-left corner
    ], dtype=np.float32)

    H = find_homography(src_pts, dst_pts)
    print("Homography Matrix:\n", H)
    
    results = warpPerspective(input_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0), H, (A4_HEIGHT_PIXELS, A4_WIDTH_PIXELS,))
    return results

def gradio_demo(image):
    result_image = predict_homography(image)
    return result_image

# Create the Gradio interface with updated syntax
iface = gr.Interface(
    fn=gradio_demo,
    inputs=gr.Image(type="numpy", label="Input Image"),  # Updated syntax
    outputs=gr.Image(type="numpy", label="Warped Output"),  # Updated syntax
    title="Point Estimation for Homography",
    description="Upload an image to estimate the homography using the PointEstimatorCNN model."
)

# Launch the Gradio interface
iface.launch(server_name="0.0.0.0", server_port=7860, share=True)