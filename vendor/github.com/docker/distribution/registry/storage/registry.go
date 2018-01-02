package storage

import (
	"regexp"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/storage/cache"
	storagedriver "github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/libtrust"
)

// registry is the top-level implementation of Registry for use in the storage
// package. All instances should descend from this object.
type registry struct {
	blobStore                    *blobStore
	blobServer                   *blobServer
	statter                      *blobStatter // global statter service.
	blobDescriptorCacheProvider  cache.BlobDescriptorCacheProvider
	deleteEnabled                bool
	resumableDigestEnabled       bool
	schema1SigningKey            libtrust.PrivateKey
	blobDescriptorServiceFactory distribution.BlobDescriptorServiceFactory
	manifestURLs                 manifestURLs
}

// manifestURLs holds regular expressions for controlling manifest URL whitelisting
type manifestURLs struct {
	allow *regexp.Regexp
	deny  *regexp.Regexp
}

// RegistryOption is the type used for functional options for NewRegistry.
type RegistryOption func(*registry) error

// EnableRedirect is a functional option for NewRegistry. It causes the backend
// blob server to attempt using (StorageDriver).URLFor to serve all blobs.
func EnableRedirect(registry *registry) error {
	registry.blobServer.redirect = true
	return nil
}

// EnableDelete is a functional option for NewRegistry. It enables deletion on
// the registry.
func EnableDelete(registry *registry) error {
	registry.deleteEnabled = true
	return nil
}

// DisableDigestResumption is a functional option for NewRegistry. It should be
// used if the registry is acting as a caching proxy.
func DisableDigestResumption(registry *registry) error {
	registry.resumableDigestEnabled = false
	return nil
}

// ManifestURLsAllowRegexp is a functional option for NewRegistry.
func ManifestURLsAllowRegexp(r *regexp.Regexp) RegistryOption {
	return func(registry *registry) error {
		registry.manifestURLs.allow = r
		return nil
	}
}

// ManifestURLsDenyRegexp is a functional option for NewRegistry.
func ManifestURLsDenyRegexp(r *regexp.Regexp) RegistryOption {
	return func(registry *registry) error {
		registry.manifestURLs.deny = r
		return nil
	}
}

// Schema1SigningKey returns a functional option for NewRegistry. It sets the
// key for signing  all schema1 manifests.
func Schema1SigningKey(key libtrust.PrivateKey) RegistryOption {
	return func(registry *registry) error {
		registry.schema1SigningKey = key
		return nil
	}
}

// BlobDescriptorServiceFactory returns a functional option for NewRegistry. It sets the
// factory to create BlobDescriptorServiceFactory middleware.
func BlobDescriptorServiceFactory(factory distribution.BlobDescriptorServiceFactory) RegistryOption {
	return func(registry *registry) error {
		registry.blobDescriptorServiceFactory = factory
		return nil
	}
}

// BlobDescriptorCacheProvider returns a functional option for
// NewRegistry. It creates a cached blob statter for use by the
// registry.
func BlobDescriptorCacheProvider(blobDescriptorCacheProvider cache.BlobDescriptorCacheProvider) RegistryOption {
	// TODO(aaronl): The duplication of statter across several objects is
	// ugly, and prevents us from using interface types in the registry
	// struct. Ideally, blobStore and blobServer should be lazily
	// initialized, and use the current value of
	// blobDescriptorCacheProvider.
	return func(registry *registry) error {
		if blobDescriptorCacheProvider != nil {
			statter := cache.NewCachedBlobStatter(blobDescriptorCacheProvider, registry.statter)
			registry.blobStore.statter = statter
			registry.blobServer.statter = statter
			registry.blobDescriptorCacheProvider = blobDescriptorCacheProvider
		}
		return nil
	}
}

// NewRegistry creates a new registry instance from the provided driver. The
// resulting registry may be shared by multiple goroutines but is cheap to
// allocate. If the Redirect option is specified, the backend blob server will
// attempt to use (StorageDriver).URLFor to serve all blobs.
func NewRegistry(ctx context.Context, driver storagedriver.StorageDriver, options ...RegistryOption) (distribution.Namespace, error) {
	// create global statter
	statter := &blobStatter{
		driver: driver,
	}

	bs := &blobStore{
		driver:  driver,
		statter: statter,
	}

	registry := &registry{
		blobStore: bs,
		blobServer: &blobServer{
			driver:  driver,
			statter: statter,
			pathFn:  bs.path,
		},
		statter:                statter,
		resumableDigestEnabled: true,
	}

	for _, option := range options {
		if err := option(registry); err != nil {
			return nil, err
		}
	}

	return registry, nil
}

// Scope returns the namespace scope for a registry. The registry
// will only serve repositories contained within this scope.
func (reg *registry) Scope() distribution.Scope {
	return distribution.GlobalScope
}

// Repository returns an instance of the repository tied to the registry.
// Instances should not be shared between goroutines but are cheap to
// allocate. In general, they should be request scoped.
func (reg *registry) Repository(ctx context.Context, canonicalName reference.Named) (distribution.Repository, error) {
	var descriptorCache distribution.BlobDescriptorService
	if reg.blobDescriptorCacheProvider != nil {
		var err error
		descriptorCache, err = reg.blobDescriptorCacheProvider.RepositoryScoped(canonicalName.Name())
		if err != nil {
			return nil, err
		}
	}

	return &repository{
		ctx:             ctx,
		registry:        reg,
		name:            canonicalName,
		descriptorCache: descriptorCache,
	}, nil
}

func (reg *registry) Blobs() distribution.BlobEnumerator {
	return reg.blobStore
}

func (reg *registry) BlobStatter() distribution.BlobStatter {
	return reg.statter
}

// repository provides name-scoped access to various services.
type repository struct {
	*registry
	ctx             context.Context
	name            reference.Named
	descriptorCache distribution.BlobDescriptorService
}

// Name returns the name of the repository.
func (repo *repository) Named() reference.Named {
	return repo.name
}

func (repo *repository) Tags(ctx context.Context) distribution.TagService {
	tags := &tagStore{
		repository: repo,
		blobStore:  repo.registry.blobStore,
	}

	return tags
}

// Manifests returns an instance of ManifestService. Instantiation is cheap and
// may be context sensitive in the future. The instance should be used similar
// to a request local.
func (repo *repository) Manifests(ctx context.Context, options ...distribution.ManifestServiceOption) (distribution.ManifestService, error) {
	manifestLinkPathFns := []linkPathFunc{
		// NOTE(stevvooe): Need to search through multiple locations since
		// 2.1.0 unintentionally linked into  _layers.
		manifestRevisionLinkPath,
		blobLinkPath,
	}

	manifestDirectoryPathSpec := manifestRevisionsPathSpec{name: repo.name.Name()}

	var statter distribution.BlobDescriptorService = &linkedBlobStatter{
		blobStore:   repo.blobStore,
		repository:  repo,
		linkPathFns: manifestLinkPathFns,
	}

	if repo.registry.blobDescriptorServiceFactory != nil {
		statter = repo.registry.blobDescriptorServiceFactory.BlobAccessController(statter)
	}

	blobStore := &linkedBlobStore{
		ctx:                  ctx,
		blobStore:            repo.blobStore,
		repository:           repo,
		deleteEnabled:        repo.registry.deleteEnabled,
		blobAccessController: statter,

		// TODO(stevvooe): linkPath limits this blob store to only
		// manifests. This instance cannot be used for blob checks.
		linkPathFns:           manifestLinkPathFns,
		linkDirectoryPathSpec: manifestDirectoryPathSpec,
	}

	ms := &manifestStore{
		ctx:        ctx,
		repository: repo,
		blobStore:  blobStore,
		schema1Handler: &signedManifestHandler{
			ctx:               ctx,
			schema1SigningKey: repo.schema1SigningKey,
			repository:        repo,
			blobStore:         blobStore,
		},
		schema2Handler: &schema2ManifestHandler{
			ctx:          ctx,
			repository:   repo,
			blobStore:    blobStore,
			manifestURLs: repo.registry.manifestURLs,
		},
		manifestListHandler: &manifestListHandler{
			ctx:        ctx,
			repository: repo,
			blobStore:  blobStore,
		},
	}

	// Apply options
	for _, option := range options {
		err := option.Apply(ms)
		if err != nil {
			return nil, err
		}
	}

	return ms, nil
}

// Blobs returns an instance of the BlobStore. Instantiation is cheap and
// may be context sensitive in the future. The instance should be used similar
// to a request local.
func (repo *repository) Blobs(ctx context.Context) distribution.BlobStore {
	var statter distribution.BlobDescriptorService = &linkedBlobStatter{
		blobStore:   repo.blobStore,
		repository:  repo,
		linkPathFns: []linkPathFunc{blobLinkPath},
	}

	if repo.descriptorCache != nil {
		statter = cache.NewCachedBlobStatter(repo.descriptorCache, statter)
	}

	if repo.registry.blobDescriptorServiceFactory != nil {
		statter = repo.registry.blobDescriptorServiceFactory.BlobAccessController(statter)
	}

	return &linkedBlobStore{
		registry:             repo.registry,
		blobStore:            repo.blobStore,
		blobServer:           repo.blobServer,
		blobAccessController: statter,
		repository:           repo,
		ctx:                  ctx,

		// TODO(stevvooe): linkPath limits this blob store to only layers.
		// This instance cannot be used for manifest checks.
		linkPathFns:            []linkPathFunc{blobLinkPath},
		deleteEnabled:          repo.registry.deleteEnabled,
		resumableDigestEnabled: repo.resumableDigestEnabled,
	}
}
