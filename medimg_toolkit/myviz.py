
import matplotlib.pyplot as plt


def show_side_by_side(images, titles=None, cmap='gray'):
    """並排顯示影像"""

    n = len(images)

    fig, axes = plt.subplots(
        1, n,
        figsize=(5 * n, 5)
    )

    if n == 1:
        axes = [axes]

    for i, img in enumerate(images):
        axes[i].imshow(img, cmap=cmap)
        axes[i].axis('off')

        if titles is not None:
            axes[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()


def overlay(img1, img2, alpha=0.5, cmap='gray'):
    """影像疊圖"""

    plt.figure(figsize=(6, 6))

    plt.imshow(img1, cmap=cmap)
    plt.imshow(img2, cmap=cmap, alpha=alpha)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
