package proxy

import (
	"io"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/digest"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/proxy/scheduler"
)

// todo(richardscothern): from cache control header or config file
const blobTTL = time.Duration(24 * 7 * time.Hour)

type proxyBlobStore struct {
	localStore     distribution.BlobStore
	remoteStore    distribution.BlobService
	scheduler      *scheduler.TTLExpirationScheduler
	repositoryName reference.Named
	authChallenger authChallenger
}

var _ distribution.BlobStore = &proxyBlobStore{}

// inflight tracks currently downloading blobs
var inflight = make(map[digest.Digest]struct{})

// mu protects inflight
var mu sync.Mutex

func setResponseHeaders(w http.ResponseWriter, length int64, mediaType string, digest digest.Digest) {
	w.Header().Set("Content-Length", strconv.FormatInt(length, 10))
	w.Header().Set("Content-Type", mediaType)
	w.Header().Set("Docker-Content-Digest", digest.String())
	w.Header().Set("Etag", digest.String())
}

func (pbs *proxyBlobStore) copyContent(ctx context.Context, dgst digest.Digest, writer io.Writer) (distribution.Descriptor, error) {
	desc, err := pbs.remoteStore.Stat(ctx, dgst)
	if err != nil {
		return distribution.Descriptor{}, err
	}

	if w, ok := writer.(http.ResponseWriter); ok {
		setResponseHeaders(w, desc.Size, desc.MediaType, dgst)
	}

	remoteReader, err := pbs.remoteStore.Open(ctx, dgst)
	if err != nil {
		return distribution.Descriptor{}, err
	}

	_, err = io.CopyN(writer, remoteReader, desc.Size)
	if err != nil {
		return distribution.Descriptor{}, err
	}

	proxyMetrics.BlobPush(uint64(desc.Size))

	return desc, nil
}

func (pbs *proxyBlobStore) serveLocal(ctx context.Context, w http.ResponseWriter, r *http.Request, dgst digest.Digest) (bool, error) {
	localDesc, err := pbs.localStore.Stat(ctx, dgst)
	if err != nil {
		// Stat can report a zero sized file here if it's checked between creation
		// and population.  Return nil error, and continue
		return false, nil
	}

	if err == nil {
		proxyMetrics.BlobPush(uint64(localDesc.Size))
		return true, pbs.localStore.ServeBlob(ctx, w, r, dgst)
	}

	return false, nil

}

func (pbs *proxyBlobStore) storeLocal(ctx context.Context, dgst digest.Digest) error {
	defer func() {
		mu.Lock()
		delete(inflight, dgst)
		mu.Unlock()
	}()

	var desc distribution.Descriptor
	var err error
	var bw distribution.BlobWriter

	bw, err = pbs.localStore.Create(ctx)
	if err != nil {
		return err
	}

	desc, err = pbs.copyContent(ctx, dgst, bw)
	if err != nil {
		return err
	}

	_, err = bw.Commit(ctx, desc)
	if err != nil {
		return err
	}

	return nil
}

func (pbs *proxyBlobStore) ServeBlob(ctx context.Context, w http.ResponseWriter, r *http.Request, dgst digest.Digest) error {
	served, err := pbs.serveLocal(ctx, w, r, dgst)
	if err != nil {
		context.GetLogger(ctx).Errorf("Error serving blob from local storage: %s", err.Error())
		return err
	}

	if served {
		return nil
	}

	if err := pbs.authChallenger.tryEstablishChallenges(ctx); err != nil {
		return err
	}

	mu.Lock()
	_, ok := inflight[dgst]
	if ok {
		mu.Unlock()
		_, err := pbs.copyContent(ctx, dgst, w)
		return err
	}
	inflight[dgst] = struct{}{}
	mu.Unlock()

	go func(dgst digest.Digest) {
		if err := pbs.storeLocal(ctx, dgst); err != nil {
			context.GetLogger(ctx).Errorf("Error committing to storage: %s", err.Error())
		}

		blobRef, err := reference.WithDigest(pbs.repositoryName, dgst)
		if err != nil {
			context.GetLogger(ctx).Errorf("Error creating reference: %s", err)
			return
		}

		pbs.scheduler.AddBlob(blobRef, repositoryTTL)
	}(dgst)

	_, err = pbs.copyContent(ctx, dgst, w)
	if err != nil {
		return err
	}
	return nil
}

func (pbs *proxyBlobStore) Stat(ctx context.Context, dgst digest.Digest) (distribution.Descriptor, error) {
	desc, err := pbs.localStore.Stat(ctx, dgst)
	if err == nil {
		return desc, err
	}

	if err != distribution.ErrBlobUnknown {
		return distribution.Descriptor{}, err
	}

	if err := pbs.authChallenger.tryEstablishChallenges(ctx); err != nil {
		return distribution.Descriptor{}, err
	}

	return pbs.remoteStore.Stat(ctx, dgst)
}

func (pbs *proxyBlobStore) Get(ctx context.Context, dgst digest.Digest) ([]byte, error) {
	blob, err := pbs.localStore.Get(ctx, dgst)
	if err == nil {
		return blob, nil
	}

	if err := pbs.authChallenger.tryEstablishChallenges(ctx); err != nil {
		return []byte{}, err
	}

	blob, err = pbs.remoteStore.Get(ctx, dgst)
	if err != nil {
		return []byte{}, err
	}

	_, err = pbs.localStore.Put(ctx, "", blob)
	if err != nil {
		return []byte{}, err
	}
	return blob, nil
}

// Unsupported functions
func (pbs *proxyBlobStore) Put(ctx context.Context, mediaType string, p []byte) (distribution.Descriptor, error) {
	return distribution.Descriptor{}, distribution.ErrUnsupported
}

func (pbs *proxyBlobStore) Create(ctx context.Context, options ...distribution.BlobCreateOption) (distribution.BlobWriter, error) {
	return nil, distribution.ErrUnsupported
}

func (pbs *proxyBlobStore) Resume(ctx context.Context, id string) (distribution.BlobWriter, error) {
	return nil, distribution.ErrUnsupported
}

func (pbs *proxyBlobStore) Mount(ctx context.Context, sourceRepo reference.Named, dgst digest.Digest) (distribution.Descriptor, error) {
	return distribution.Descriptor{}, distribution.ErrUnsupported
}

func (pbs *proxyBlobStore) Open(ctx context.Context, dgst digest.Digest) (distribution.ReadSeekCloser, error) {
	return nil, distribution.ErrUnsupported
}

func (pbs *proxyBlobStore) Delete(ctx context.Context, dgst digest.Digest) error {
	return distribution.ErrUnsupported
}
