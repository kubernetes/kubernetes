package memory

import (
	"sync"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/digest"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/storage/cache"
)

type inMemoryBlobDescriptorCacheProvider struct {
	global       *mapBlobDescriptorCache
	repositories map[string]*mapBlobDescriptorCache
	mu           sync.RWMutex
}

// NewInMemoryBlobDescriptorCacheProvider returns a new mapped-based cache for
// storing blob descriptor data.
func NewInMemoryBlobDescriptorCacheProvider() cache.BlobDescriptorCacheProvider {
	return &inMemoryBlobDescriptorCacheProvider{
		global:       newMapBlobDescriptorCache(),
		repositories: make(map[string]*mapBlobDescriptorCache),
	}
}

func (imbdcp *inMemoryBlobDescriptorCacheProvider) RepositoryScoped(repo string) (distribution.BlobDescriptorService, error) {
	if _, err := reference.ParseNamed(repo); err != nil {
		return nil, err
	}

	imbdcp.mu.RLock()
	defer imbdcp.mu.RUnlock()

	return &repositoryScopedInMemoryBlobDescriptorCache{
		repo:       repo,
		parent:     imbdcp,
		repository: imbdcp.repositories[repo],
	}, nil
}

func (imbdcp *inMemoryBlobDescriptorCacheProvider) Stat(ctx context.Context, dgst digest.Digest) (distribution.Descriptor, error) {
	return imbdcp.global.Stat(ctx, dgst)
}

func (imbdcp *inMemoryBlobDescriptorCacheProvider) Clear(ctx context.Context, dgst digest.Digest) error {
	return imbdcp.global.Clear(ctx, dgst)
}

func (imbdcp *inMemoryBlobDescriptorCacheProvider) SetDescriptor(ctx context.Context, dgst digest.Digest, desc distribution.Descriptor) error {
	_, err := imbdcp.Stat(ctx, dgst)
	if err == distribution.ErrBlobUnknown {

		if dgst.Algorithm() != desc.Digest.Algorithm() && dgst != desc.Digest {
			// if the digests differ, set the other canonical mapping
			if err := imbdcp.global.SetDescriptor(ctx, desc.Digest, desc); err != nil {
				return err
			}
		}

		// unknown, just set it
		return imbdcp.global.SetDescriptor(ctx, dgst, desc)
	}

	// we already know it, do nothing
	return err
}

// repositoryScopedInMemoryBlobDescriptorCache provides the request scoped
// repository cache. Instances are not thread-safe but the delegated
// operations are.
type repositoryScopedInMemoryBlobDescriptorCache struct {
	repo       string
	parent     *inMemoryBlobDescriptorCacheProvider // allows lazy allocation of repo's map
	repository *mapBlobDescriptorCache
}

func (rsimbdcp *repositoryScopedInMemoryBlobDescriptorCache) Stat(ctx context.Context, dgst digest.Digest) (distribution.Descriptor, error) {
	if rsimbdcp.repository == nil {
		return distribution.Descriptor{}, distribution.ErrBlobUnknown
	}

	return rsimbdcp.repository.Stat(ctx, dgst)
}

func (rsimbdcp *repositoryScopedInMemoryBlobDescriptorCache) Clear(ctx context.Context, dgst digest.Digest) error {
	if rsimbdcp.repository == nil {
		return distribution.ErrBlobUnknown
	}

	return rsimbdcp.repository.Clear(ctx, dgst)
}

func (rsimbdcp *repositoryScopedInMemoryBlobDescriptorCache) SetDescriptor(ctx context.Context, dgst digest.Digest, desc distribution.Descriptor) error {
	if rsimbdcp.repository == nil {
		// allocate map since we are setting it now.
		rsimbdcp.parent.mu.Lock()
		var ok bool
		// have to read back value since we may have allocated elsewhere.
		rsimbdcp.repository, ok = rsimbdcp.parent.repositories[rsimbdcp.repo]
		if !ok {
			rsimbdcp.repository = newMapBlobDescriptorCache()
			rsimbdcp.parent.repositories[rsimbdcp.repo] = rsimbdcp.repository
		}

		rsimbdcp.parent.mu.Unlock()
	}

	if err := rsimbdcp.repository.SetDescriptor(ctx, dgst, desc); err != nil {
		return err
	}

	return rsimbdcp.parent.SetDescriptor(ctx, dgst, desc)
}

// mapBlobDescriptorCache provides a simple map-based implementation of the
// descriptor cache.
type mapBlobDescriptorCache struct {
	descriptors map[digest.Digest]distribution.Descriptor
	mu          sync.RWMutex
}

var _ distribution.BlobDescriptorService = &mapBlobDescriptorCache{}

func newMapBlobDescriptorCache() *mapBlobDescriptorCache {
	return &mapBlobDescriptorCache{
		descriptors: make(map[digest.Digest]distribution.Descriptor),
	}
}

func (mbdc *mapBlobDescriptorCache) Stat(ctx context.Context, dgst digest.Digest) (distribution.Descriptor, error) {
	if err := dgst.Validate(); err != nil {
		return distribution.Descriptor{}, err
	}

	mbdc.mu.RLock()
	defer mbdc.mu.RUnlock()

	desc, ok := mbdc.descriptors[dgst]
	if !ok {
		return distribution.Descriptor{}, distribution.ErrBlobUnknown
	}

	return desc, nil
}

func (mbdc *mapBlobDescriptorCache) Clear(ctx context.Context, dgst digest.Digest) error {
	mbdc.mu.Lock()
	defer mbdc.mu.Unlock()

	delete(mbdc.descriptors, dgst)
	return nil
}

func (mbdc *mapBlobDescriptorCache) SetDescriptor(ctx context.Context, dgst digest.Digest, desc distribution.Descriptor) error {
	if err := dgst.Validate(); err != nil {
		return err
	}

	if err := cache.ValidateDescriptor(desc); err != nil {
		return err
	}

	mbdc.mu.Lock()
	defer mbdc.mu.Unlock()

	mbdc.descriptors[dgst] = desc
	return nil
}
