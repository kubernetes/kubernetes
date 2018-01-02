package distribution

import (
	"encoding/json"
	"fmt"
	"io"
	"runtime"

	"github.com/docker/distribution"
	"github.com/docker/distribution/manifest/schema2"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/distribution/metadata"
	"github.com/docker/docker/distribution/xfer"
	"github.com/docker/docker/image"
	"github.com/docker/docker/layer"
	"github.com/docker/docker/pkg/progress"
	"github.com/docker/docker/pkg/system"
	refstore "github.com/docker/docker/reference"
	"github.com/docker/docker/registry"
	"github.com/docker/libtrust"
	"github.com/opencontainers/go-digest"
	"golang.org/x/net/context"
)

// Config stores configuration for communicating
// with a registry.
type Config struct {
	// MetaHeaders stores HTTP headers with metadata about the image
	MetaHeaders map[string][]string
	// AuthConfig holds authentication credentials for authenticating with
	// the registry.
	AuthConfig *types.AuthConfig
	// ProgressOutput is the interface for showing the status of the pull
	// operation.
	ProgressOutput progress.Output
	// RegistryService is the registry service to use for TLS configuration
	// and endpoint lookup.
	RegistryService registry.Service
	// ImageEventLogger notifies events for a given image
	ImageEventLogger func(id, name, action string)
	// MetadataStore is the storage backend for distribution-specific
	// metadata.
	MetadataStore metadata.Store
	// ImageStore manages images.
	ImageStore ImageConfigStore
	// ReferenceStore manages tags. This value is optional, when excluded
	// content will not be tagged.
	ReferenceStore refstore.Store
	// RequireSchema2 ensures that only schema2 manifests are used.
	RequireSchema2 bool
}

// ImagePullConfig stores pull configuration.
type ImagePullConfig struct {
	Config

	// DownloadManager manages concurrent pulls.
	DownloadManager RootFSDownloadManager
	// Schema2Types is the valid schema2 configuration types allowed
	// by the pull operation.
	Schema2Types []string
	// Platform is the requested platform of the image being pulled to ensure it can be validated
	// when the host platform supports multiple image operating systems.
	Platform string
}

// ImagePushConfig stores push configuration.
type ImagePushConfig struct {
	Config

	// ConfigMediaType is the configuration media type for
	// schema2 manifests.
	ConfigMediaType string
	// LayerStore manages layers.
	LayerStore PushLayerProvider
	// TrustKey is the private key for legacy signatures. This is typically
	// an ephemeral key, since these signatures are no longer verified.
	TrustKey libtrust.PrivateKey
	// UploadManager dispatches uploads.
	UploadManager *xfer.LayerUploadManager
}

// ImageConfigStore handles storing and getting image configurations
// by digest. Allows getting an image configurations rootfs from the
// configuration.
type ImageConfigStore interface {
	Put([]byte) (digest.Digest, error)
	Get(digest.Digest) ([]byte, error)
	RootFSAndPlatformFromConfig([]byte) (*image.RootFS, layer.Platform, error)
}

// PushLayerProvider provides layers to be pushed by ChainID.
type PushLayerProvider interface {
	Get(layer.ChainID) (PushLayer, error)
}

// PushLayer is a pushable layer with metadata about the layer
// and access to the content of the layer.
type PushLayer interface {
	ChainID() layer.ChainID
	DiffID() layer.DiffID
	Parent() PushLayer
	Open() (io.ReadCloser, error)
	Size() (int64, error)
	MediaType() string
	Release()
}

// RootFSDownloadManager handles downloading of the rootfs
type RootFSDownloadManager interface {
	// Download downloads the layers into the given initial rootfs and
	// returns the final rootfs.
	// Given progress output to track download progress
	// Returns function to release download resources
	Download(ctx context.Context, initialRootFS image.RootFS, platform layer.Platform, layers []xfer.DownloadDescriptor, progressOutput progress.Output) (image.RootFS, func(), error)
}

type imageConfigStore struct {
	image.Store
}

// NewImageConfigStoreFromStore returns an ImageConfigStore backed
// by an image.Store for container images.
func NewImageConfigStoreFromStore(is image.Store) ImageConfigStore {
	return &imageConfigStore{
		Store: is,
	}
}

func (s *imageConfigStore) Put(c []byte) (digest.Digest, error) {
	id, err := s.Store.Create(c)
	return digest.Digest(id), err
}

func (s *imageConfigStore) Get(d digest.Digest) ([]byte, error) {
	img, err := s.Store.Get(image.IDFromDigest(d))
	if err != nil {
		return nil, err
	}
	return img.RawJSON(), nil
}

func (s *imageConfigStore) RootFSAndPlatformFromConfig(c []byte) (*image.RootFS, layer.Platform, error) {
	var unmarshalledConfig image.Image
	if err := json.Unmarshal(c, &unmarshalledConfig); err != nil {
		return nil, "", err
	}

	// fail immediately on Windows when downloading a non-Windows image
	// and vice versa. Exception on Windows if Linux Containers are enabled.
	if runtime.GOOS == "windows" && unmarshalledConfig.OS == "linux" && !system.LCOWSupported() {
		return nil, "", fmt.Errorf("image operating system %q cannot be used on this platform", unmarshalledConfig.OS)
	} else if runtime.GOOS != "windows" && unmarshalledConfig.OS == "windows" {
		return nil, "", fmt.Errorf("image operating system %q cannot be used on this platform", unmarshalledConfig.OS)
	}

	platform := ""
	if runtime.GOOS == "windows" {
		platform = unmarshalledConfig.OS
	}
	return unmarshalledConfig.RootFS, layer.Platform(platform), nil
}

type storeLayerProvider struct {
	ls layer.Store
}

// NewLayerProviderFromStore returns a layer provider backed by
// an instance of LayerStore. Only getting layers as gzipped
// tars is supported.
func NewLayerProviderFromStore(ls layer.Store) PushLayerProvider {
	return &storeLayerProvider{
		ls: ls,
	}
}

func (p *storeLayerProvider) Get(lid layer.ChainID) (PushLayer, error) {
	if lid == "" {
		return &storeLayer{
			Layer: layer.EmptyLayer,
		}, nil
	}
	l, err := p.ls.Get(lid)
	if err != nil {
		return nil, err
	}

	sl := storeLayer{
		Layer: l,
		ls:    p.ls,
	}
	if d, ok := l.(distribution.Describable); ok {
		return &describableStoreLayer{
			storeLayer:  sl,
			describable: d,
		}, nil
	}

	return &sl, nil
}

type storeLayer struct {
	layer.Layer
	ls layer.Store
}

func (l *storeLayer) Parent() PushLayer {
	p := l.Layer.Parent()
	if p == nil {
		return nil
	}
	sl := storeLayer{
		Layer: p,
		ls:    l.ls,
	}
	if d, ok := p.(distribution.Describable); ok {
		return &describableStoreLayer{
			storeLayer:  sl,
			describable: d,
		}
	}

	return &sl
}

func (l *storeLayer) Open() (io.ReadCloser, error) {
	return l.Layer.TarStream()
}

func (l *storeLayer) Size() (int64, error) {
	return l.Layer.DiffSize()
}

func (l *storeLayer) MediaType() string {
	// layer store always returns uncompressed tars
	return schema2.MediaTypeUncompressedLayer
}

func (l *storeLayer) Release() {
	if l.ls != nil {
		layer.ReleaseAndLog(l.ls, l.Layer)
	}
}

type describableStoreLayer struct {
	storeLayer
	describable distribution.Describable
}

func (l *describableStoreLayer) Descriptor() distribution.Descriptor {
	return l.describable.Descriptor()
}
