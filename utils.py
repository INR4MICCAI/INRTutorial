from typing import Optional, List, Tuple
import torch
from matplotlib import pyplot as plt


def min_max_normalize(image: torch.Tensor, min: Optional[float] = None, max: Optional[float] = None):
    if min is None:
        min = image.min()
    if max is None:
        max = image.max()
    image_norm = (image - min) / (max - min)
    return image_norm


def ncc(template: torch.Tensor, image: torch.Tensor):
    """ Normalized cross-correlation """
    # Compute means and standard deviatons
    template_mean = template.mean()
    image_mean = image.mean()
    template_std = template.std()
    image_std = image.std()

    # Compute correlation and normalize
    correlation = ((template - template_mean) * (image - image_mean)).mean()
    ncc = correlation / (template_std * image_std)
    return ncc


def psnr(pred, ref):
    max_value = ref.max()
    mse = torch.mean((pred - ref) ** 2, dim=(-2, -1))
    out = 20 * torch.log10(max_value / torch.sqrt(mse))
    return out.mean()


def plot_reconstructions(progress_ims: List[Tuple[int, torch.Tensor]], gt_im: torch.Tensor):
    ncols = len(progress_ims) + 1
    fig_width = 5
    fig, axs = plt.subplots(ncols=ncols, figsize=(ncols*fig_width, fig_width))
    # Plot all reconstructions images predicted by the model
    for i, (epoch, im, metric) in enumerate(progress_ims):
        im = im.cpu().numpy()
        ax = axs[i]
        ax.imshow(im, cmap='gray')
        ax.axis('off')
        title = f'Epoch: {epoch},\n PSNR: {metric}'
        ax.set_title(title)
    # PLot ground-truth image
    gt_im = gt_im.cpu().numpy()
    axs[-1].imshow(gt_im, cmap='gray')
    axs[-1].axis('off')
    axs[-1].set_title('Ground Truth')
    plt.tight_layout()
    plt.show()
