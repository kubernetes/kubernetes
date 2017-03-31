package content

import (
	"sync"

	"github.com/nightlyone/lockfile"
	"github.com/pkg/errors"
)

// In addition to providing inter-process locks for content ingest, we also
// define a global in process lock to prevent two goroutines writing to the
// same file.
//
// This is pretty unsophisticated for now. In the future, we'd probably like to
// have more information about who is holding which locks, as well as better
// error reporting.

var (
	errLocked = errors.New("key is locked")

	// locks lets us lock in process, as well as output of process.
	locks   = map[lockfile.Lockfile]struct{}{}
	locksMu sync.Mutex
)

func tryLock(lock lockfile.Lockfile) error {
	locksMu.Lock()
	defer locksMu.Unlock()

	if _, ok := locks[lock]; ok {
		return errLocked
	}

	if err := lock.TryLock(); err != nil {
		if errors.Cause(err) == lockfile.ErrBusy {
			return errLocked
		}

		return errors.Wrapf(err, "lock.TryLock() encountered an error")
	}

	locks[lock] = struct{}{}
	return nil
}

func unlock(lock lockfile.Lockfile) error {
	locksMu.Lock()
	defer locksMu.Unlock()

	if _, ok := locks[lock]; !ok {
		return nil
	}

	delete(locks, lock)
	return lock.Unlock()
}
