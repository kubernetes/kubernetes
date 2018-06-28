// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"

	"github.com/pkg/errors"
	"golang.org/x/sync/errgroup"
)

// A Solution is returned by a solver run. It is mostly just a Lock, with some
// additional methods that report information about the solve run.
type Solution interface {
	Lock
	// The name of the ProjectAnalyzer used in generating this solution.
	AnalyzerName() string
	// The version of the ProjectAnalyzer used in generating this solution.
	AnalyzerVersion() int
	// The name of the Solver used in generating this solution.
	SolverName() string
	// The version of the Solver used in generating this solution.
	SolverVersion() int
	Attempts() int
}

type solution struct {
	// A list of the projects selected by the solver.
	p []LockedProject

	// The number of solutions that were attempted
	att int

	// The hash digest of the input opts
	hd []byte

	// The analyzer info
	analyzerInfo ProjectAnalyzerInfo

	// The solver used in producing this solution
	solv Solver
}

const concurrentWriters = 16

// WriteDepTree takes a basedir, a Lock and a RootPruneOptions and exports all
// the projects listed in the lock to the appropriate target location within basedir.
//
// If the goal is to populate a vendor directory, basedir should be the absolute
// path to that vendor directory, not its parent (a project root, typically).
//
// It requires a SourceManager to do the work. Prune options are read from the
// passed manifest.
func WriteDepTree(basedir string, l Lock, sm SourceManager, co CascadingPruneOptions, logger *log.Logger) error {
	if l == nil {
		return fmt.Errorf("must provide non-nil Lock to WriteDepTree")
	}

	if err := os.MkdirAll(basedir, 0777); err != nil {
		return err
	}

	g, ctx := errgroup.WithContext(context.TODO())
	lps := l.Projects()
	sem := make(chan struct{}, concurrentWriters)
	var cnt struct {
		sync.Mutex
		i int
	}

	for i := range lps {
		p := lps[i] // per-iteration copy

		g.Go(func() error {
			err := func() error {
				select {
				case sem <- struct{}{}:
					defer func() { <-sem }()
				case <-ctx.Done():
					return ctx.Err()
				}

				ident := p.Ident()
				projectRoot := string(ident.ProjectRoot)
				to := filepath.FromSlash(filepath.Join(basedir, projectRoot))

				if err := sm.ExportProject(ctx, ident, p.Version(), to); err != nil {
					return errors.Wrapf(err, "failed to export %s", projectRoot)
				}

				err := PruneProject(to, p, co.PruneOptionsFor(ident.ProjectRoot), logger)
				if err != nil {
					return errors.Wrapf(err, "failed to prune %s", projectRoot)
				}

				return ctx.Err()
			}()

			switch err {
			case context.Canceled, context.DeadlineExceeded:
				// Don't log "secondary" errors.
			default:
				msg := "Wrote"
				if err != nil {
					msg = "Failed to write"
				}

				// Log and increment atomically to prevent re-ordering.
				cnt.Lock()
				cnt.i++
				logger.Printf("(%d/%d) %s %s@%s\n", cnt.i, len(lps), msg, p.Ident(), p.Version())
				cnt.Unlock()
			}

			return err
		})
	}

	err := g.Wait()
	if err != nil {
		os.RemoveAll(basedir)
	}
	return errors.Wrap(err, "failed to write dep tree")
}

func (r solution) Projects() []LockedProject {
	return r.p
}

func (r solution) Attempts() int {
	return r.att
}

func (r solution) InputsDigest() []byte {
	return r.hd
}

func (r solution) AnalyzerName() string {
	return r.analyzerInfo.Name
}

func (r solution) AnalyzerVersion() int {
	return r.analyzerInfo.Version
}

func (r solution) SolverName() string {
	return r.solv.Name()
}

func (r solution) SolverVersion() int {
	return r.solv.Version()
}
