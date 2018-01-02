package notifications

import (
	"net/http"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/reference"
	"github.com/opencontainers/go-digest"
)

// ManifestListener describes a set of methods for listening to events related to manifests.
type ManifestListener interface {
	ManifestPushed(repo reference.Named, sm distribution.Manifest, options ...distribution.ManifestServiceOption) error
	ManifestPulled(repo reference.Named, sm distribution.Manifest, options ...distribution.ManifestServiceOption) error
	ManifestDeleted(repo reference.Named, dgst digest.Digest) error
}

// BlobListener describes a listener that can respond to layer related events.
type BlobListener interface {
	BlobPushed(repo reference.Named, desc distribution.Descriptor) error
	BlobPulled(repo reference.Named, desc distribution.Descriptor) error
	BlobMounted(repo reference.Named, desc distribution.Descriptor, fromRepo reference.Named) error
	BlobDeleted(repo reference.Named, desc digest.Digest) error
}

// Listener combines all repository events into a single interface.
type Listener interface {
	ManifestListener
	BlobListener
}

type repositoryListener struct {
	distribution.Repository
	listener Listener
}

// Listen dispatches events on the repository to the listener.
func Listen(repo distribution.Repository, listener Listener) distribution.Repository {
	return &repositoryListener{
		Repository: repo,
		listener:   listener,
	}
}

func (rl *repositoryListener) Manifests(ctx context.Context, options ...distribution.ManifestServiceOption) (distribution.ManifestService, error) {
	manifests, err := rl.Repository.Manifests(ctx, options...)
	if err != nil {
		return nil, err
	}
	return &manifestServiceListener{
		ManifestService: manifests,
		parent:          rl,
	}, nil
}

func (rl *repositoryListener) Blobs(ctx context.Context) distribution.BlobStore {
	return &blobServiceListener{
		BlobStore: rl.Repository.Blobs(ctx),
		parent:    rl,
	}
}

type manifestServiceListener struct {
	distribution.ManifestService
	parent *repositoryListener
}

func (msl *manifestServiceListener) Delete(ctx context.Context, dgst digest.Digest) error {
	err := msl.ManifestService.Delete(ctx, dgst)
	if err == nil {
		if err := msl.parent.listener.ManifestDeleted(msl.parent.Repository.Named(), dgst); err != nil {
			context.GetLogger(ctx).Errorf("error dispatching manifest delete to listener: %v", err)
		}
	}

	return err
}

func (msl *manifestServiceListener) Get(ctx context.Context, dgst digest.Digest, options ...distribution.ManifestServiceOption) (distribution.Manifest, error) {
	sm, err := msl.ManifestService.Get(ctx, dgst, options...)
	if err == nil {
		if err := msl.parent.listener.ManifestPulled(msl.parent.Repository.Named(), sm, options...); err != nil {
			context.GetLogger(ctx).Errorf("error dispatching manifest pull to listener: %v", err)
		}
	}

	return sm, err
}

func (msl *manifestServiceListener) Put(ctx context.Context, sm distribution.Manifest, options ...distribution.ManifestServiceOption) (digest.Digest, error) {
	dgst, err := msl.ManifestService.Put(ctx, sm, options...)

	if err == nil {
		if err := msl.parent.listener.ManifestPushed(msl.parent.Repository.Named(), sm, options...); err != nil {
			context.GetLogger(ctx).Errorf("error dispatching manifest push to listener: %v", err)
		}
	}

	return dgst, err
}

type blobServiceListener struct {
	distribution.BlobStore
	parent *repositoryListener
}

var _ distribution.BlobStore = &blobServiceListener{}

func (bsl *blobServiceListener) Get(ctx context.Context, dgst digest.Digest) ([]byte, error) {
	p, err := bsl.BlobStore.Get(ctx, dgst)
	if err == nil {
		if desc, err := bsl.Stat(ctx, dgst); err != nil {
			context.GetLogger(ctx).Errorf("error resolving descriptor in ServeBlob listener: %v", err)
		} else {
			if err := bsl.parent.listener.BlobPulled(bsl.parent.Repository.Named(), desc); err != nil {
				context.GetLogger(ctx).Errorf("error dispatching layer pull to listener: %v", err)
			}
		}
	}

	return p, err
}

func (bsl *blobServiceListener) Open(ctx context.Context, dgst digest.Digest) (distribution.ReadSeekCloser, error) {
	rc, err := bsl.BlobStore.Open(ctx, dgst)
	if err == nil {
		if desc, err := bsl.Stat(ctx, dgst); err != nil {
			context.GetLogger(ctx).Errorf("error resolving descriptor in ServeBlob listener: %v", err)
		} else {
			if err := bsl.parent.listener.BlobPulled(bsl.parent.Repository.Named(), desc); err != nil {
				context.GetLogger(ctx).Errorf("error dispatching layer pull to listener: %v", err)
			}
		}
	}

	return rc, err
}

func (bsl *blobServiceListener) ServeBlob(ctx context.Context, w http.ResponseWriter, r *http.Request, dgst digest.Digest) error {
	err := bsl.BlobStore.ServeBlob(ctx, w, r, dgst)
	if err == nil {
		if desc, err := bsl.Stat(ctx, dgst); err != nil {
			context.GetLogger(ctx).Errorf("error resolving descriptor in ServeBlob listener: %v", err)
		} else {
			if err := bsl.parent.listener.BlobPulled(bsl.parent.Repository.Named(), desc); err != nil {
				context.GetLogger(ctx).Errorf("error dispatching layer pull to listener: %v", err)
			}
		}
	}

	return err
}

func (bsl *blobServiceListener) Put(ctx context.Context, mediaType string, p []byte) (distribution.Descriptor, error) {
	desc, err := bsl.BlobStore.Put(ctx, mediaType, p)
	if err == nil {
		if err := bsl.parent.listener.BlobPushed(bsl.parent.Repository.Named(), desc); err != nil {
			context.GetLogger(ctx).Errorf("error dispatching layer push to listener: %v", err)
		}
	}

	return desc, err
}

func (bsl *blobServiceListener) Create(ctx context.Context, options ...distribution.BlobCreateOption) (distribution.BlobWriter, error) {
	wr, err := bsl.BlobStore.Create(ctx, options...)
	switch err := err.(type) {
	case distribution.ErrBlobMounted:
		if err := bsl.parent.listener.BlobMounted(bsl.parent.Repository.Named(), err.Descriptor, err.From); err != nil {
			context.GetLogger(ctx).Errorf("error dispatching blob mount to listener: %v", err)
		}
		return nil, err
	}
	return bsl.decorateWriter(wr), err
}

func (bsl *blobServiceListener) Delete(ctx context.Context, dgst digest.Digest) error {
	err := bsl.BlobStore.Delete(ctx, dgst)
	if err == nil {
		if err := bsl.parent.listener.BlobDeleted(bsl.parent.Repository.Named(), dgst); err != nil {
			context.GetLogger(ctx).Errorf("error dispatching layer delete to listener: %v", err)
		}
	}

	return err
}

func (bsl *blobServiceListener) Resume(ctx context.Context, id string) (distribution.BlobWriter, error) {
	wr, err := bsl.BlobStore.Resume(ctx, id)
	return bsl.decorateWriter(wr), err
}

func (bsl *blobServiceListener) decorateWriter(wr distribution.BlobWriter) distribution.BlobWriter {
	return &blobWriterListener{
		BlobWriter: wr,
		parent:     bsl,
	}
}

type blobWriterListener struct {
	distribution.BlobWriter
	parent *blobServiceListener
}

func (bwl *blobWriterListener) Commit(ctx context.Context, desc distribution.Descriptor) (distribution.Descriptor, error) {
	committed, err := bwl.BlobWriter.Commit(ctx, desc)
	if err == nil {
		if err := bwl.parent.parent.listener.BlobPushed(bwl.parent.parent.Repository.Named(), committed); err != nil {
			context.GetLogger(ctx).Errorf("error dispatching blob push to listener: %v", err)
		}
	}

	return committed, err
}
