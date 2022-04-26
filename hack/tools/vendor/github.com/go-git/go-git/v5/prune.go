package git

import (
	"errors"
	"time"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/storer"
)

type PruneHandler func(unreferencedObjectHash plumbing.Hash) error
type PruneOptions struct {
	// OnlyObjectsOlderThan if set to non-zero value
	// selects only objects older than the time provided.
	OnlyObjectsOlderThan time.Time
	// Handler is called on matching objects
	Handler PruneHandler
}

var ErrLooseObjectsNotSupported = errors.New("Loose objects not supported")

// DeleteObject deletes an object from a repository.
// The type conveniently matches PruneHandler.
func (r *Repository) DeleteObject(hash plumbing.Hash) error {
	los, ok := r.Storer.(storer.LooseObjectStorer)
	if !ok {
		return ErrLooseObjectsNotSupported
	}

	return los.DeleteLooseObject(hash)
}

func (r *Repository) Prune(opt PruneOptions) error {
	los, ok := r.Storer.(storer.LooseObjectStorer)
	if !ok {
		return ErrLooseObjectsNotSupported
	}

	pw := newObjectWalker(r.Storer)
	err := pw.walkAllRefs()
	if err != nil {
		return err
	}
	// Now walk all (loose) objects in storage.
	return los.ForEachObjectHash(func(hash plumbing.Hash) error {
		// Get out if we have seen this object.
		if pw.isSeen(hash) {
			return nil
		}
		// Otherwise it is a candidate for pruning.
		// Check out for too new objects next.
		if !opt.OnlyObjectsOlderThan.IsZero() {
			// Errors here are non-fatal. The object may be e.g. packed.
			// Or concurrently deleted. Skip such objects.
			t, err := los.LooseObjectTime(hash)
			if err != nil {
				return nil
			}
			// Skip too new objects.
			if !t.Before(opt.OnlyObjectsOlderThan) {
				return nil
			}
		}
		return opt.Handler(hash)
	})
}
