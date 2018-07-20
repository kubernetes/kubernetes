package daemon

import (
	"io"
	"runtime"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/builder"
	"github.com/docker/docker/image"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/containerfs"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/registry"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

type releaseableLayer struct {
	released   bool
	layerStore layer.Store
	roLayer    layer.Layer
	rwLayer    layer.RWLayer
}

func (rl *releaseableLayer) Mount() (containerfs.ContainerFS, error) {
	var err error
	var mountPath containerfs.ContainerFS
	var chainID layer.ChainID
	if rl.roLayer != nil {
		chainID = rl.roLayer.ChainID()
	}

	mountID := stringid.GenerateRandomID()
	rl.rwLayer, err = rl.layerStore.CreateRWLayer(mountID, chainID, nil)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create rwlayer")
	}

	mountPath, err = rl.rwLayer.Mount("")
	if err != nil {
		// Clean up the layer if we fail to mount it here.
		metadata, err := rl.layerStore.ReleaseRWLayer(rl.rwLayer)
		layer.LogReleaseMetadata(metadata)
		if err != nil {
			logrus.Errorf("Failed to release RWLayer: %s", err)
		}
		rl.rwLayer = nil
		return nil, err
	}

	return mountPath, nil
}

func (rl *releaseableLayer) Commit(os string) (builder.ReleaseableLayer, error) {
	var chainID layer.ChainID
	if rl.roLayer != nil {
		chainID = rl.roLayer.ChainID()
	}

	stream, err := rl.rwLayer.TarStream()
	if err != nil {
		return nil, err
	}
	defer stream.Close()

	newLayer, err := rl.layerStore.Register(stream, chainID, layer.OS(os))
	if err != nil {
		return nil, err
	}
	// TODO: An optimization woudld be to handle empty layers before returning
	return &releaseableLayer{layerStore: rl.layerStore, roLayer: newLayer}, nil
}

func (rl *releaseableLayer) DiffID() layer.DiffID {
	if rl.roLayer == nil {
		return layer.DigestSHA256EmptyTar
	}
	return rl.roLayer.DiffID()
}

func (rl *releaseableLayer) Release() error {
	if rl.released {
		return nil
	}
	if err := rl.releaseRWLayer(); err != nil {
		// Best effort attempt at releasing read-only layer before returning original error.
		rl.releaseROLayer()
		return err
	}
	if err := rl.releaseROLayer(); err != nil {
		return err
	}
	rl.released = true
	return nil
}

func (rl *releaseableLayer) releaseRWLayer() error {
	if rl.rwLayer == nil {
		return nil
	}
	if err := rl.rwLayer.Unmount(); err != nil {
		logrus.Errorf("Failed to unmount RWLayer: %s", err)
		return err
	}
	metadata, err := rl.layerStore.ReleaseRWLayer(rl.rwLayer)
	layer.LogReleaseMetadata(metadata)
	if err != nil {
		logrus.Errorf("Failed to release RWLayer: %s", err)
	}
	rl.rwLayer = nil
	return err
}

func (rl *releaseableLayer) releaseROLayer() error {
	if rl.roLayer == nil {
		return nil
	}
	metadata, err := rl.layerStore.Release(rl.roLayer)
	layer.LogReleaseMetadata(metadata)
	if err != nil {
		logrus.Errorf("Failed to release ROLayer: %s", err)
	}
	rl.roLayer = nil
	return err
}

func newReleasableLayerForImage(img *image.Image, layerStore layer.Store) (builder.ReleaseableLayer, error) {
	if img == nil || img.RootFS.ChainID() == "" {
		return &releaseableLayer{layerStore: layerStore}, nil
	}
	// Hold a reference to the image layer so that it can't be removed before
	// it is released
	roLayer, err := layerStore.Get(img.RootFS.ChainID())
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get layer for image %s", img.ImageID())
	}
	return &releaseableLayer{layerStore: layerStore, roLayer: roLayer}, nil
}

// TODO: could this use the regular daemon PullImage ?
func (daemon *Daemon) pullForBuilder(ctx context.Context, name string, authConfigs map[string]types.AuthConfig, output io.Writer, platform string) (*image.Image, error) {
	ref, err := reference.ParseNormalizedNamed(name)
	if err != nil {
		return nil, err
	}
	ref = reference.TagNameOnly(ref)

	pullRegistryAuth := &types.AuthConfig{}
	if len(authConfigs) > 0 {
		// The request came with a full auth config, use it
		repoInfo, err := daemon.RegistryService.ResolveRepository(ref)
		if err != nil {
			return nil, err
		}

		resolvedConfig := registry.ResolveAuthConfig(authConfigs, repoInfo.Index)
		pullRegistryAuth = &resolvedConfig
	}

	if err := daemon.pullImageWithReference(ctx, ref, platform, nil, pullRegistryAuth, output); err != nil {
		return nil, err
	}
	return daemon.GetImage(name)
}

// GetImageAndReleasableLayer returns an image and releaseable layer for a reference or ID.
// Every call to GetImageAndReleasableLayer MUST call releasableLayer.Release() to prevent
// leaking of layers.
func (daemon *Daemon) GetImageAndReleasableLayer(ctx context.Context, refOrID string, opts backend.GetImageAndLayerOptions) (builder.Image, builder.ReleaseableLayer, error) {
	if refOrID == "" {
		layer, err := newReleasableLayerForImage(nil, daemon.stores[opts.OS].layerStore)
		return nil, layer, err
	}

	if opts.PullOption != backend.PullOptionForcePull {
		image, err := daemon.GetImage(refOrID)
		if err != nil && opts.PullOption == backend.PullOptionNoPull {
			return nil, nil, err
		}
		// TODO: shouldn't we error out if error is different from "not found" ?
		if image != nil {
			layer, err := newReleasableLayerForImage(image, daemon.stores[opts.OS].layerStore)
			return image, layer, err
		}
	}

	image, err := daemon.pullForBuilder(ctx, refOrID, opts.AuthConfig, opts.Output, opts.OS)
	if err != nil {
		return nil, nil, err
	}
	layer, err := newReleasableLayerForImage(image, daemon.stores[opts.OS].layerStore)
	return image, layer, err
}

// CreateImage creates a new image by adding a config and ID to the image store.
// This is similar to LoadImage() except that it receives JSON encoded bytes of
// an image instead of a tar archive.
func (daemon *Daemon) CreateImage(config []byte, parent string, platform string) (builder.Image, error) {
	if platform == "" {
		platform = runtime.GOOS
	}
	id, err := daemon.stores[platform].imageStore.Create(config)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to create image")
	}

	if parent != "" {
		if err := daemon.stores[platform].imageStore.SetParent(id, image.ID(parent)); err != nil {
			return nil, errors.Wrapf(err, "failed to set parent %s", parent)
		}
	}

	return daemon.stores[platform].imageStore.Get(id)
}

// IDMappings returns uid/gid mappings for the builder
func (daemon *Daemon) IDMappings() *idtools.IDMappings {
	return daemon.idMappings
}
