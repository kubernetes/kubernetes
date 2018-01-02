package images

import (
	imagesapi "github.com/containerd/containerd/api/services/images/v1"
	"github.com/containerd/containerd/api/types"
	"github.com/containerd/containerd/images"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
)

func imagesToProto(images []images.Image) []imagesapi.Image {
	var imagespb []imagesapi.Image

	for _, image := range images {
		imagespb = append(imagespb, imageToProto(&image))
	}

	return imagespb
}

func imagesFromProto(imagespb []imagesapi.Image) []images.Image {
	var images []images.Image

	for _, image := range imagespb {
		images = append(images, imageFromProto(&image))
	}

	return images
}

func imageToProto(image *images.Image) imagesapi.Image {
	return imagesapi.Image{
		Name:      image.Name,
		Labels:    image.Labels,
		Target:    descToProto(&image.Target),
		CreatedAt: image.CreatedAt,
		UpdatedAt: image.UpdatedAt,
	}
}

func imageFromProto(imagepb *imagesapi.Image) images.Image {
	return images.Image{
		Name:      imagepb.Name,
		Labels:    imagepb.Labels,
		Target:    descFromProto(&imagepb.Target),
		CreatedAt: imagepb.CreatedAt,
		UpdatedAt: imagepb.UpdatedAt,
	}
}

func descFromProto(desc *types.Descriptor) ocispec.Descriptor {
	return ocispec.Descriptor{
		MediaType: desc.MediaType,
		Size:      desc.Size_,
		Digest:    desc.Digest,
	}
}

func descToProto(desc *ocispec.Descriptor) types.Descriptor {
	return types.Descriptor{
		MediaType: desc.MediaType,
		Size_:     desc.Size,
		Digest:    desc.Digest,
	}
}
