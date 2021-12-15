package pwalk

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
)

type WalkFunc = filepath.WalkFunc

// Walk is a wrapper for filepath.Walk which can call multiple walkFn
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
func Walk(root string, walkFn WalkFunc) error {
	return WalkN(root, walkFn, runtime.NumCPU()*2)
}

// WalkN is a wrapper for filepath.Walk which can call multiple walkFn
// in parallel, allowing to handle each item concurrently. A maximum of
// num walkFn will be called at any one time.
//
// Please see Walk documentation for caveats of using this function.
func WalkN(root string, walkFn WalkFunc, num int) error {
	// make sure limit is sensible
	if num < 1 {
		return fmt.Errorf("walk(%q): num must be > 0", root)
	}

	files := make(chan *walkArgs, 2*num)
	errCh := make(chan error, 1) // get the first error, ignore others

	// Start walking a tree asap
	var (
		err error
		wg  sync.WaitGroup

		rootLen   = len(root)
		rootEntry *walkArgs
	)
	wg.Add(1)
	go func() {
		err = filepath.Walk(root, func(p string, info os.FileInfo, err error) error {
			if err != nil {
				close(files)
				return err
			}
			if len(p) == rootLen {
				// Root entry is processed separately below.
				rootEntry = &walkArgs{path: p, info: &info}
				return nil
			}
			// add a file to the queue unless a callback sent an error
			select {
			case e := <-errCh:
				close(files)
				return e
			default:
				files <- &walkArgs{path: p, info: &info}
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
				if e := walkFn(file.path, *file.info, nil); e != nil {
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
		err = walkFn(rootEntry.path, *rootEntry.info, nil)
	}

	return err
}

// walkArgs holds the arguments that were passed to the Walk or WalkN
// functions.
type walkArgs struct {
	path string
	info *os.FileInfo
}
