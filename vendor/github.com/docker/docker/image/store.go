package image

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/docker/distribution/digestset"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/system"
	"github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// Store is an interface for creating and accessing images
type Store interface {
	Create(config []byte) (ID, error)
	Get(id ID) (*Image, error)
	Delete(id ID) ([]layer.Metadata, error)
	Search(partialID string) (ID, error)
	SetParent(id ID, parent ID) error
	GetParent(id ID) (ID, error)
	SetLastUpdated(id ID) error
	GetLastUpdated(id ID) (time.Time, error)
	Children(id ID) []ID
	Map() map[ID]*Image
	Heads() map[ID]*Image
}

// LayerGetReleaser is a minimal interface for getting and releasing images.
type LayerGetReleaser interface {
	Get(layer.ChainID) (layer.Layer, error)
	Release(layer.Layer) ([]layer.Metadata, error)
}

type imageMeta struct {
	layer    layer.Layer
	children map[ID]struct{}
}

type store struct {
	sync.RWMutex
	ls        LayerGetReleaser
	images    map[ID]*imageMeta
	fs        StoreBackend
	digestSet *digestset.Set
	platform  string
}

// NewImageStore returns new store object for given layer store
func NewImageStore(fs StoreBackend, platform string, ls LayerGetReleaser) (Store, error) {
	is := &store{
		ls:        ls,
		images:    make(map[ID]*imageMeta),
		fs:        fs,
		digestSet: digestset.NewSet(),
		platform:  platform,
	}

	// load all current images and retain layers
	if err := is.restore(); err != nil {
		return nil, err
	}

	return is, nil
}

func (is *store) restore() error {
	err := is.fs.Walk(func(dgst digest.Digest) error {
		img, err := is.Get(IDFromDigest(dgst))
		if err != nil {
			logrus.Errorf("invalid image %v, %v", dgst, err)
			return nil
		}
		var l layer.Layer
		if chainID := img.RootFS.ChainID(); chainID != "" {
			l, err = is.ls.Get(chainID)
			if err != nil {
				return err
			}
		}
		if err := is.digestSet.Add(dgst); err != nil {
			return err
		}

		imageMeta := &imageMeta{
			layer:    l,
			children: make(map[ID]struct{}),
		}

		is.images[IDFromDigest(dgst)] = imageMeta

		return nil
	})
	if err != nil {
		return err
	}

	// Second pass to fill in children maps
	for id := range is.images {
		if parent, err := is.GetParent(id); err == nil {
			if parentMeta := is.images[parent]; parentMeta != nil {
				parentMeta.children[id] = struct{}{}
			}
		}
	}

	return nil
}

func (is *store) Create(config []byte) (ID, error) {
	var img Image
	err := json.Unmarshal(config, &img)
	if err != nil {
		return "", err
	}

	// TODO @jhowardmsft - LCOW Support. This will need revisiting.
	// Integrity check - ensure we are creating something for the correct platform
	if system.LCOWSupported() {
		if strings.ToLower(img.Platform()) != strings.ToLower(is.platform) {
			return "", fmt.Errorf("cannot create entry for platform %q in image store for platform %q", img.Platform(), is.platform)
		}
	}

	// Must reject any config that references diffIDs from the history
	// which aren't among the rootfs layers.
	rootFSLayers := make(map[layer.DiffID]struct{})
	for _, diffID := range img.RootFS.DiffIDs {
		rootFSLayers[diffID] = struct{}{}
	}

	layerCounter := 0
	for _, h := range img.History {
		if !h.EmptyLayer {
			layerCounter++
		}
	}
	if layerCounter > len(img.RootFS.DiffIDs) {
		return "", errors.New("too many non-empty layers in History section")
	}

	dgst, err := is.fs.Set(config)
	if err != nil {
		return "", err
	}
	imageID := IDFromDigest(dgst)

	is.Lock()
	defer is.Unlock()

	if _, exists := is.images[imageID]; exists {
		return imageID, nil
	}

	layerID := img.RootFS.ChainID()

	var l layer.Layer
	if layerID != "" {
		l, err = is.ls.Get(layerID)
		if err != nil {
			return "", errors.Wrapf(err, "failed to get layer %s", layerID)
		}
	}

	imageMeta := &imageMeta{
		layer:    l,
		children: make(map[ID]struct{}),
	}

	is.images[imageID] = imageMeta
	if err := is.digestSet.Add(imageID.Digest()); err != nil {
		delete(is.images, imageID)
		return "", err
	}

	return imageID, nil
}

func (is *store) Search(term string) (ID, error) {
	dgst, err := is.digestSet.Lookup(term)
	if err != nil {
		if err == digestset.ErrDigestNotFound {
			err = fmt.Errorf("No such image: %s", term)
		}
		return "", err
	}
	return IDFromDigest(dgst), nil
}

func (is *store) Get(id ID) (*Image, error) {
	// todo: Check if image is in images
	// todo: Detect manual insertions and start using them
	config, err := is.fs.Get(id.Digest())
	if err != nil {
		return nil, err
	}

	img, err := NewFromJSON(config)
	if err != nil {
		return nil, err
	}
	img.computedID = id

	img.Parent, err = is.GetParent(id)
	if err != nil {
		img.Parent = ""
	}

	return img, nil
}

func (is *store) Delete(id ID) ([]layer.Metadata, error) {
	is.Lock()
	defer is.Unlock()

	imageMeta := is.images[id]
	if imageMeta == nil {
		return nil, fmt.Errorf("unrecognized image ID %s", id.String())
	}
	for id := range imageMeta.children {
		is.fs.DeleteMetadata(id.Digest(), "parent")
	}
	if parent, err := is.GetParent(id); err == nil && is.images[parent] != nil {
		delete(is.images[parent].children, id)
	}

	if err := is.digestSet.Remove(id.Digest()); err != nil {
		logrus.Errorf("error removing %s from digest set: %q", id, err)
	}
	delete(is.images, id)
	is.fs.Delete(id.Digest())

	if imageMeta.layer != nil {
		return is.ls.Release(imageMeta.layer)
	}
	return nil, nil
}

func (is *store) SetParent(id, parent ID) error {
	is.Lock()
	defer is.Unlock()
	parentMeta := is.images[parent]
	if parentMeta == nil {
		return fmt.Errorf("unknown parent image ID %s", parent.String())
	}
	if parent, err := is.GetParent(id); err == nil && is.images[parent] != nil {
		delete(is.images[parent].children, id)
	}
	parentMeta.children[id] = struct{}{}
	return is.fs.SetMetadata(id.Digest(), "parent", []byte(parent))
}

func (is *store) GetParent(id ID) (ID, error) {
	d, err := is.fs.GetMetadata(id.Digest(), "parent")
	if err != nil {
		return "", err
	}
	return ID(d), nil // todo: validate?
}

// SetLastUpdated time for the image ID to the current time
func (is *store) SetLastUpdated(id ID) error {
	lastUpdated := []byte(time.Now().Format(time.RFC3339Nano))
	return is.fs.SetMetadata(id.Digest(), "lastUpdated", lastUpdated)
}

// GetLastUpdated time for the image ID
func (is *store) GetLastUpdated(id ID) (time.Time, error) {
	bytes, err := is.fs.GetMetadata(id.Digest(), "lastUpdated")
	if err != nil || len(bytes) == 0 {
		// No lastUpdated time
		return time.Time{}, nil
	}
	return time.Parse(time.RFC3339Nano, string(bytes))
}

func (is *store) Children(id ID) []ID {
	is.RLock()
	defer is.RUnlock()

	return is.children(id)
}

func (is *store) children(id ID) []ID {
	var ids []ID
	if is.images[id] != nil {
		for id := range is.images[id].children {
			ids = append(ids, id)
		}
	}
	return ids
}

func (is *store) Heads() map[ID]*Image {
	return is.imagesMap(false)
}

func (is *store) Map() map[ID]*Image {
	return is.imagesMap(true)
}

func (is *store) imagesMap(all bool) map[ID]*Image {
	is.RLock()
	defer is.RUnlock()

	images := make(map[ID]*Image)

	for id := range is.images {
		if !all && len(is.children(id)) > 0 {
			continue
		}
		img, err := is.Get(id)
		if err != nil {
			logrus.Errorf("invalid image access: %q, error: %q", id, err)
			continue
		}
		images[id] = img
	}
	return images
}
