package storage

import (
	"path"
	"sync"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/digest"
)

type signatureStore struct {
	repository *repository
	blobStore  *blobStore
	ctx        context.Context
}

func (s *signatureStore) Get(dgst digest.Digest) ([][]byte, error) {
	signaturesPath, err := pathFor(manifestSignaturesPathSpec{
		name:     s.repository.Named().Name(),
		revision: dgst,
	})

	if err != nil {
		return nil, err
	}

	// Need to append signature digest algorithm to path to get all items.
	// Perhaps, this should be in the pathMapper but it feels awkward. This
	// can be eliminated by implementing listAll on drivers.
	signaturesPath = path.Join(signaturesPath, "sha256")

	signaturePaths, err := s.blobStore.driver.List(s.ctx, signaturesPath)
	if err != nil {
		return nil, err
	}

	var wg sync.WaitGroup
	type result struct {
		index     int
		signature []byte
		err       error
	}
	ch := make(chan result)

	bs := s.linkedBlobStore(s.ctx, dgst)
	for i, sigPath := range signaturePaths {
		sigdgst, err := digest.ParseDigest("sha256:" + path.Base(sigPath))
		if err != nil {
			context.GetLogger(s.ctx).Errorf("could not get digest from path: %q, skipping", sigPath)
			continue
		}

		wg.Add(1)
		go func(idx int, sigdgst digest.Digest) {
			defer wg.Done()
			context.GetLogger(s.ctx).
				Debugf("fetching signature %q", sigdgst)

			r := result{index: idx}

			if p, err := bs.Get(s.ctx, sigdgst); err != nil {
				context.GetLogger(s.ctx).
					Errorf("error fetching signature %q: %v", sigdgst, err)
				r.err = err
			} else {
				r.signature = p
			}

			ch <- r
		}(i, sigdgst)
	}
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	// aggregrate the results
	signatures := make([][]byte, len(signaturePaths))
loop:
	for {
		select {
		case result := <-ch:
			signatures[result.index] = result.signature
			if result.err != nil && err == nil {
				// only set the first one.
				err = result.err
			}
		case <-done:
			break loop
		}
	}

	return signatures, err
}

func (s *signatureStore) Put(dgst digest.Digest, signatures ...[]byte) error {
	bs := s.linkedBlobStore(s.ctx, dgst)
	for _, signature := range signatures {
		if _, err := bs.Put(s.ctx, "application/json", signature); err != nil {
			return err
		}
	}
	return nil
}

// linkedBlobStore returns the namedBlobStore of the signatures for the
// manifest with the given digest. Effectively, each signature link path
// layout is a unique linked blob store.
func (s *signatureStore) linkedBlobStore(ctx context.Context, revision digest.Digest) *linkedBlobStore {
	linkpath := func(name string, dgst digest.Digest) (string, error) {
		return pathFor(manifestSignatureLinkPathSpec{
			name:      name,
			revision:  revision,
			signature: dgst,
		})

	}

	return &linkedBlobStore{
		ctx:        ctx,
		repository: s.repository,
		blobStore:  s.blobStore,
		blobAccessController: &linkedBlobStatter{
			blobStore:   s.blobStore,
			repository:  s.repository,
			linkPathFns: []linkPathFunc{linkpath},
		},
		linkPathFns: []linkPathFunc{linkpath},
	}
}
