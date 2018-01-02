package images

import (
	"context"

	imagesapi "github.com/containerd/containerd/api/services/images/v1"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/images"
	ptypes "github.com/gogo/protobuf/types"
)

type remoteStore struct {
	client imagesapi.ImagesClient
}

// NewStoreFromClient returns a new image store client
func NewStoreFromClient(client imagesapi.ImagesClient) images.Store {
	return &remoteStore{
		client: client,
	}
}

func (s *remoteStore) Get(ctx context.Context, name string) (images.Image, error) {
	resp, err := s.client.Get(ctx, &imagesapi.GetImageRequest{
		Name: name,
	})
	if err != nil {
		return images.Image{}, errdefs.FromGRPC(err)
	}

	return imageFromProto(resp.Image), nil
}

func (s *remoteStore) List(ctx context.Context, filters ...string) ([]images.Image, error) {
	resp, err := s.client.List(ctx, &imagesapi.ListImagesRequest{
		Filters: filters,
	})
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}

	return imagesFromProto(resp.Images), nil
}

func (s *remoteStore) Create(ctx context.Context, image images.Image) (images.Image, error) {
	created, err := s.client.Create(ctx, &imagesapi.CreateImageRequest{
		Image: imageToProto(&image),
	})
	if err != nil {
		return images.Image{}, errdefs.FromGRPC(err)
	}

	return imageFromProto(&created.Image), nil
}

func (s *remoteStore) Update(ctx context.Context, image images.Image, fieldpaths ...string) (images.Image, error) {
	var updateMask *ptypes.FieldMask
	if len(fieldpaths) > 0 {
		updateMask = &ptypes.FieldMask{
			Paths: fieldpaths,
		}
	}

	updated, err := s.client.Update(ctx, &imagesapi.UpdateImageRequest{
		Image:      imageToProto(&image),
		UpdateMask: updateMask,
	})
	if err != nil {
		return images.Image{}, errdefs.FromGRPC(err)
	}

	return imageFromProto(&updated.Image), nil
}

func (s *remoteStore) Delete(ctx context.Context, name string) error {
	_, err := s.client.Delete(ctx, &imagesapi.DeleteImageRequest{
		Name: name,
	})

	return errdefs.FromGRPC(err)
}
