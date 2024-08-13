//go:build go1.16
// +build go1.16

package pwalkdir

import (
	"fmt"
	"io/fs"
	"path/filepath"
	"runtime"
	"sync"
)

// Walk is a wrapper for filepath.WalkDir which can call multiple walkFn
// in parallel, allowing to handle each item concurrently. A maximum of
// twice the runtime.NumCPU() walkFn will be called at any one time.
// If you want to change the maximum, use WalkN instead.
//
// The order of calls is non-deterministic.
//
// Note that this implementation only supports primitive error handling:
//
// - no errors are ever passed to walkFn;
//
// - once a walkFn returns any error, all further processing stops
// and the error is returned to the caller of Walk;
//
// - filepath.SkipDir is not supported;
//
// - if more than one walkFn instance will return an error, only one
// of such errors will be propagated and returned by Walk, others
// will be silently discarded.
func Walk(root string, walkFn fs.WalkDirFunc) error {
	return WalkN(root, walkFn, runtime.NumCPU()*2)
}

// WalkN is a wrapper for filepath.WalkDir which can call multiple walkFn
// in parallel, allowing to handle each item concurrently. A maximum of
// num walkFn will be called at any one time.
//
// Please see Walk documentation for caveats of using this function.
func WalkN(root string, walkFn fs.WalkDirFunc, num int) error {
	// make sure limit is sensible
	if num < 1 {
		return fmt.Errorf("walk(%q): num must be > 0", root)
	}

	files := make(chan *walkArgs, 2*num)
	errCh := make(chan error, 1) // Get the first error, ignore others.

	// Start walking a tree asap.
	var (
		err error
		wg  sync.WaitGroup

		rootLen   = len(root)
		rootEntry *walkArgs
	)
	wg.Add(1)
	go func() {
		err = filepath.WalkDir(root, func(p string, entry fs.DirEntry, err error) error {
			if err != nil {
				close(files)
				return err
			}
			if len(p) == rootLen {
				// Root entry is processed separately below.
				rootEntry = &walkArgs{path: p, entry: entry}
				return nil
			}
			// Add a file to the queue unless a callback sent an error.
			select {
			case e := <-errCh:
				close(files)
				return e
			default:
				files <- &walkArgs{path: p, entry: entry}
				return nil
			}
		})
		if err == nil {
			close(files)
		}
		wg.Done()
	}()

	wg.Add(num)
	for i := 0; i < num; i++ {
		go func() {
			for file := range files {
				if e := walkFn(file.path, file.entry, nil); e != nil {
					select {
					case errCh <- e: // sent ok
					default: // buffer full
					}
				}
			}
			wg.Done()
		}()
	}

	wg.Wait()

	if err == nil {
		err = walkFn(rootEntry.path, rootEntry.entry, nil)
	}

	return err
}

// walkArgs holds the arguments that were passed to the Walk or WalkN
// functions.
type walkArgs struct {
	entry fs.DirEntry
	path  string
}
