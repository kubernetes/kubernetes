package main

import (
	"archive/tar"
	"bytes"
	"context"
	"io"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/api/types/volume"
	"github.com/docker/docker/client"
)

func createTar(data map[string][]byte) (io.Reader, error) {
	var b bytes.Buffer
	tw := tar.NewWriter(&b)
	for path, datum := range data {
		hdr := tar.Header{
			Name: path,
			Mode: 0644,
			Size: int64(len(datum)),
		}
		if err := tw.WriteHeader(&hdr); err != nil {
			return nil, err
		}
		_, err := tw.Write(datum)
		if err != nil {
			return nil, err
		}
	}
	if err := tw.Close(); err != nil {
		return nil, err
	}
	return &b, nil
}

// createVolumeWithData creates a volume with the given data (e.g. data["/foo"] = []byte("bar"))
// Internally, a container is created from the image so as to provision the data to the volume,
// which is attached to the container.
func createVolumeWithData(cli *client.Client, volumeName string, data map[string][]byte, image string) error {
	_, err := cli.VolumeCreate(context.Background(),
		volume.VolumesCreateBody{
			Driver: "local",
			Name:   volumeName,
		})
	if err != nil {
		return err
	}
	mnt := "/mnt"
	miniContainer, err := cli.ContainerCreate(context.Background(),
		&container.Config{
			Image: image,
		},
		&container.HostConfig{
			Mounts: []mount.Mount{
				{
					Type:   mount.TypeVolume,
					Source: volumeName,
					Target: mnt,
				},
			},
		}, nil, "")
	if err != nil {
		return err
	}
	tr, err := createTar(data)
	if err != nil {
		return err
	}
	if cli.CopyToContainer(context.Background(),
		miniContainer.ID, mnt, tr, types.CopyToContainerOptions{}); err != nil {
		return err
	}
	return cli.ContainerRemove(context.Background(),
		miniContainer.ID,
		types.ContainerRemoveOptions{})
}

func hasVolume(cli *client.Client, volumeName string) bool {
	_, err := cli.VolumeInspect(context.Background(), volumeName)
	return err == nil
}

func removeVolume(cli *client.Client, volumeName string) error {
	return cli.VolumeRemove(context.Background(), volumeName, true)
}
