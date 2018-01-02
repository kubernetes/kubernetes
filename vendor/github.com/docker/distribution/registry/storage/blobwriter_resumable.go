// +build !noresumabledigest

package storage

import (
	"fmt"
	"path"
	"strconv"

	"github.com/docker/distribution/context"
	storagedriver "github.com/docker/distribution/registry/storage/driver"
	"github.com/sirupsen/logrus"
	"github.com/stevvooe/resumable"

	// register resumable hashes with import
	_ "github.com/stevvooe/resumable/sha256"
	_ "github.com/stevvooe/resumable/sha512"
)

// resumeDigest attempts to restore the state of the internal hash function
// by loading the most recent saved hash state equal to the current size of the blob.
func (bw *blobWriter) resumeDigest(ctx context.Context) error {
	if !bw.resumableDigestEnabled {
		return errResumableDigestNotAvailable
	}

	h, ok := bw.digester.Hash().(resumable.Hash)
	if !ok {
		return errResumableDigestNotAvailable
	}
	offset := bw.fileWriter.Size()
	if offset == int64(h.Len()) {
		// State of digester is already at the requested offset.
		return nil
	}

	// List hash states from storage backend.
	var hashStateMatch hashStateEntry
	hashStates, err := bw.getStoredHashStates(ctx)
	if err != nil {
		return fmt.Errorf("unable to get stored hash states with offset %d: %s", offset, err)
	}

	// Find the highest stored hashState with offset equal to
	// the requested offset.
	for _, hashState := range hashStates {
		if hashState.offset == offset {
			hashStateMatch = hashState
			break // Found an exact offset match.
		}
	}

	if hashStateMatch.offset == 0 {
		// No need to load any state, just reset the hasher.
		h.Reset()
	} else {
		storedState, err := bw.driver.GetContent(ctx, hashStateMatch.path)
		if err != nil {
			return err
		}

		if err = h.Restore(storedState); err != nil {
			return err
		}
	}

	// Mind the gap.
	if gapLen := offset - int64(h.Len()); gapLen > 0 {
		return errResumableDigestNotAvailable
	}

	return nil
}

type hashStateEntry struct {
	offset int64
	path   string
}

// getStoredHashStates returns a slice of hashStateEntries for this upload.
func (bw *blobWriter) getStoredHashStates(ctx context.Context) ([]hashStateEntry, error) {
	uploadHashStatePathPrefix, err := pathFor(uploadHashStatePathSpec{
		name: bw.blobStore.repository.Named().String(),
		id:   bw.id,
		alg:  bw.digester.Digest().Algorithm(),
		list: true,
	})

	if err != nil {
		return nil, err
	}

	paths, err := bw.blobStore.driver.List(ctx, uploadHashStatePathPrefix)
	if err != nil {
		if _, ok := err.(storagedriver.PathNotFoundError); !ok {
			return nil, err
		}
		// Treat PathNotFoundError as no entries.
		paths = nil
	}

	hashStateEntries := make([]hashStateEntry, 0, len(paths))

	for _, p := range paths {
		pathSuffix := path.Base(p)
		// The suffix should be the offset.
		offset, err := strconv.ParseInt(pathSuffix, 0, 64)
		if err != nil {
			logrus.Errorf("unable to parse offset from upload state path %q: %s", p, err)
		}

		hashStateEntries = append(hashStateEntries, hashStateEntry{offset: offset, path: p})
	}

	return hashStateEntries, nil
}

func (bw *blobWriter) storeHashState(ctx context.Context) error {
	if !bw.resumableDigestEnabled {
		return errResumableDigestNotAvailable
	}

	h, ok := bw.digester.Hash().(resumable.Hash)
	if !ok {
		return errResumableDigestNotAvailable
	}

	uploadHashStatePath, err := pathFor(uploadHashStatePathSpec{
		name:   bw.blobStore.repository.Named().String(),
		id:     bw.id,
		alg:    bw.digester.Digest().Algorithm(),
		offset: int64(h.Len()),
	})

	if err != nil {
		return err
	}

	hashState, err := h.State()
	if err != nil {
		return err
	}

	return bw.driver.PutContent(ctx, uploadHashStatePath, hashState)
}
