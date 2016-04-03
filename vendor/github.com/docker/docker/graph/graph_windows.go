// +build windows

package graph

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/graphdriver/windows"
	"github.com/docker/docker/image"
	"github.com/docker/docker/pkg/archive"
)

// setupInitLayer populates a directory with mountpoints suitable
// for bind-mounting dockerinit into the container. T
func SetupInitLayer(initLayer string) error {
	return nil
}

func createRootFilesystemInDriver(graph *Graph, img *image.Image, layerData archive.ArchiveReader) error {
	if wd, ok := graph.driver.(*windows.WindowsGraphDriver); ok {
		if img.Container != "" && layerData == nil {
			logrus.Debugf("Copying from container %s.", img.Container)

			var ids []string
			if img.Parent != "" {
				parentImg, err := graph.Get(img.Parent)
				if err != nil {
					return err
				}

				ids, err = graph.ParentLayerIds(parentImg)
				if err != nil {
					return err
				}
			}

			if err := wd.CopyDiff(img.Container, img.ID, wd.LayerIdsToPaths(ids)); err != nil {
				return fmt.Errorf("Driver %s failed to copy image rootfs %s: %s", graph.driver, img.Container, err)
			}
		} else if img.Parent == "" {
			if err := graph.driver.Create(img.ID, img.Parent); err != nil {
				return fmt.Errorf("Driver %s failed to create image rootfs %s: %s", graph.driver, img.ID, err)
			}
		}
	} else {
		// This fallback allows the use of VFS during daemon development.
		if err := graph.driver.Create(img.ID, img.Parent); err != nil {
			return fmt.Errorf("Driver %s failed to create image rootfs %s: %s", graph.driver, img.ID, err)
		}
	}
	return nil
}

func (graph *Graph) restoreBaseImages() ([]string, error) {
	// TODO Windows. This needs implementing (@swernli)
	return nil, nil
}

// ParentLayerIds returns a list of all parent image IDs for the given image.
func (graph *Graph) ParentLayerIds(img *image.Image) (ids []string, err error) {
	for i := img; i != nil && err == nil; i, err = graph.GetParent(i) {
		ids = append(ids, i.ID)
	}

	return
}

// storeImage stores file system layer data for the given image to the
// graph's storage driver. Image metadata is stored in a file
// at the specified root directory.
func (graph *Graph) storeImage(img *image.Image, layerData archive.ArchiveReader, root string) (err error) {

	if wd, ok := graph.driver.(*windows.WindowsGraphDriver); ok {
		// Store the layer. If layerData is not nil and this isn't a base image,
		// unpack it into the new layer
		if layerData != nil && img.Parent != "" {
			var ids []string
			if img.Parent != "" {
				parentImg, err := graph.Get(img.Parent)
				if err != nil {
					return err
				}

				ids, err = graph.ParentLayerIds(parentImg)
				if err != nil {
					return err
				}
			}

			if img.Size, err = wd.Import(img.ID, layerData, wd.LayerIdsToPaths(ids)); err != nil {
				return err
			}
		}

		if err := graph.saveSize(root, int(img.Size)); err != nil {
			return err
		}

		f, err := os.OpenFile(jsonPath(root), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.FileMode(0600))
		if err != nil {
			return err
		}

		defer f.Close()

		return json.NewEncoder(f).Encode(img)
	} else {
		// We keep this functionality here so that we can still work with the
		// VFS driver during development. This will not be used for actual running
		// of Windows containers. Without this code, it would not be possible to
		// docker pull using the VFS driver.

		// Store the layer. If layerData is not nil, unpack it into the new layer
		if layerData != nil {
			if img.Size, err = graph.driver.ApplyDiff(img.ID, img.Parent, layerData); err != nil {
				return err
			}
		}

		if err := graph.saveSize(root, int(img.Size)); err != nil {
			return err
		}

		f, err := os.OpenFile(jsonPath(root), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.FileMode(0600))
		if err != nil {
			return err
		}

		defer f.Close()

		return json.NewEncoder(f).Encode(img)
	}
}

// TarLayer returns a tar archive of the image's filesystem layer.
func (graph *Graph) TarLayer(img *image.Image) (arch archive.Archive, err error) {
	if wd, ok := graph.driver.(*windows.WindowsGraphDriver); ok {
		var ids []string
		if img.Parent != "" {
			parentImg, err := graph.Get(img.Parent)
			if err != nil {
				return nil, err
			}

			ids, err = graph.ParentLayerIds(parentImg)
			if err != nil {
				return nil, err
			}
		}

		return wd.Export(img.ID, wd.LayerIdsToPaths(ids))
	} else {
		// We keep this functionality here so that we can still work with the VFS
		// driver during development. VFS is not supported (and just will not work)
		// for Windows containers.
		return graph.driver.Diff(img.ID, img.Parent)
	}
}
