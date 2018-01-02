package cache

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/dockerversion"
	"github.com/docker/docker/image"
	"github.com/docker/docker/layer"
	"github.com/pkg/errors"
)

// NewLocal returns a local image cache, based on parent chain
func NewLocal(store image.Store) *LocalImageCache {
	return &LocalImageCache{
		store: store,
	}
}

// LocalImageCache is cache based on parent chain.
type LocalImageCache struct {
	store image.Store
}

// GetCache returns the image id found in the cache
func (lic *LocalImageCache) GetCache(imgID string, config *containertypes.Config) (string, error) {
	return getImageIDAndError(getLocalCachedImage(lic.store, image.ID(imgID), config))
}

// New returns an image cache, based on history objects
func New(store image.Store) *ImageCache {
	return &ImageCache{
		store:           store,
		localImageCache: NewLocal(store),
	}
}

// ImageCache is cache based on history objects. Requires initial set of images.
type ImageCache struct {
	sources         []*image.Image
	store           image.Store
	localImageCache *LocalImageCache
}

// Populate adds an image to the cache (to be queried later)
func (ic *ImageCache) Populate(image *image.Image) {
	ic.sources = append(ic.sources, image)
}

// GetCache returns the image id found in the cache
func (ic *ImageCache) GetCache(parentID string, cfg *containertypes.Config) (string, error) {
	imgID, err := ic.localImageCache.GetCache(parentID, cfg)
	if err != nil {
		return "", err
	}
	if imgID != "" {
		for _, s := range ic.sources {
			if ic.isParent(s.ID(), image.ID(imgID)) {
				return imgID, nil
			}
		}
	}

	var parent *image.Image
	lenHistory := 0
	if parentID != "" {
		parent, err = ic.store.Get(image.ID(parentID))
		if err != nil {
			return "", errors.Wrapf(err, "unable to find image %v", parentID)
		}
		lenHistory = len(parent.History)
	}

	for _, target := range ic.sources {
		if !isValidParent(target, parent) || !isValidConfig(cfg, target.History[lenHistory]) {
			continue
		}

		if len(target.History)-1 == lenHistory { // last
			if parent != nil {
				if err := ic.store.SetParent(target.ID(), parent.ID()); err != nil {
					return "", errors.Wrapf(err, "failed to set parent for %v to %v", target.ID(), parent.ID())
				}
			}
			return target.ID().String(), nil
		}

		imgID, err := ic.restoreCachedImage(parent, target, cfg)
		if err != nil {
			return "", errors.Wrapf(err, "failed to restore cached image from %q to %v", parentID, target.ID())
		}

		ic.sources = []*image.Image{target} // avoid jumping to different target, tuned for safety atm
		return imgID.String(), nil
	}

	return "", nil
}

func (ic *ImageCache) restoreCachedImage(parent, target *image.Image, cfg *containertypes.Config) (image.ID, error) {
	var history []image.History
	rootFS := image.NewRootFS()
	lenHistory := 0
	if parent != nil {
		history = parent.History
		rootFS = parent.RootFS
		lenHistory = len(parent.History)
	}
	history = append(history, target.History[lenHistory])
	if layer := getLayerForHistoryIndex(target, lenHistory); layer != "" {
		rootFS.Append(layer)
	}

	config, err := json.Marshal(&image.Image{
		V1Image: image.V1Image{
			DockerVersion: dockerversion.Version,
			Config:        cfg,
			Architecture:  target.Architecture,
			OS:            target.OS,
			Author:        target.Author,
			Created:       history[len(history)-1].Created,
		},
		RootFS:     rootFS,
		History:    history,
		OSFeatures: target.OSFeatures,
		OSVersion:  target.OSVersion,
	})
	if err != nil {
		return "", errors.Wrap(err, "failed to marshal image config")
	}

	imgID, err := ic.store.Create(config)
	if err != nil {
		return "", errors.Wrap(err, "failed to create cache image")
	}

	if parent != nil {
		if err := ic.store.SetParent(imgID, parent.ID()); err != nil {
			return "", errors.Wrapf(err, "failed to set parent for %v to %v", target.ID(), parent.ID())
		}
	}
	return imgID, nil
}

func (ic *ImageCache) isParent(imgID, parentID image.ID) bool {
	nextParent, err := ic.store.GetParent(imgID)
	if err != nil {
		return false
	}
	if nextParent == parentID {
		return true
	}
	return ic.isParent(nextParent, parentID)
}

func getLayerForHistoryIndex(image *image.Image, index int) layer.DiffID {
	layerIndex := 0
	for i, h := range image.History {
		if i == index {
			if h.EmptyLayer {
				return ""
			}
			break
		}
		if !h.EmptyLayer {
			layerIndex++
		}
	}
	return image.RootFS.DiffIDs[layerIndex] // validate?
}

func isValidConfig(cfg *containertypes.Config, h image.History) bool {
	// todo: make this format better than join that loses data
	return strings.Join(cfg.Cmd, " ") == h.CreatedBy
}

func isValidParent(img, parent *image.Image) bool {
	if len(img.History) == 0 {
		return false
	}
	if parent == nil || len(parent.History) == 0 && len(parent.RootFS.DiffIDs) == 0 {
		return true
	}
	if len(parent.History) >= len(img.History) {
		return false
	}
	if len(parent.RootFS.DiffIDs) > len(img.RootFS.DiffIDs) {
		return false
	}

	for i, h := range parent.History {
		if !reflect.DeepEqual(h, img.History[i]) {
			return false
		}
	}
	for i, d := range parent.RootFS.DiffIDs {
		if d != img.RootFS.DiffIDs[i] {
			return false
		}
	}
	return true
}

func getImageIDAndError(img *image.Image, err error) (string, error) {
	if img == nil || err != nil {
		return "", err
	}
	return img.ID().String(), nil
}

// getLocalCachedImage returns the most recent created image that is a child
// of the image with imgID, that had the same config when it was
// created. nil is returned if a child cannot be found. An error is
// returned if the parent image cannot be found.
func getLocalCachedImage(imageStore image.Store, imgID image.ID, config *containertypes.Config) (*image.Image, error) {
	// Loop on the children of the given image and check the config
	getMatch := func(siblings []image.ID) (*image.Image, error) {
		var match *image.Image
		for _, id := range siblings {
			img, err := imageStore.Get(id)
			if err != nil {
				return nil, fmt.Errorf("unable to find image %q", id)
			}

			if compare(&img.ContainerConfig, config) {
				// check for the most up to date match
				if match == nil || match.Created.Before(img.Created) {
					match = img
				}
			}
		}
		return match, nil
	}

	// In this case, this is `FROM scratch`, which isn't an actual image.
	if imgID == "" {
		images := imageStore.Map()
		var siblings []image.ID
		for id, img := range images {
			if img.Parent == imgID {
				siblings = append(siblings, id)
			}
		}
		return getMatch(siblings)
	}

	// find match from child images
	siblings := imageStore.Children(imgID)
	return getMatch(siblings)
}
