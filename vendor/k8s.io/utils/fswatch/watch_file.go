/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package fswatch

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"time"
)

// FileOption configures WatchFile.
type FileOption func(*fileConfig)

type fileConfig struct {
	recheckInterval time.Duration
	fallbackPolling time.Duration
	initialCallback bool
	errorHandler    func(error)
}

// WithRecheckInterval makes onChange fire every d, regardless of
// filesystem events. Used by reloaders that need to retry transient
// apply failures even when the watched file is otherwise stable.
func WithRecheckInterval(d time.Duration) FileOption {
	return func(c *fileConfig) { c.recheckInterval = d }
}

// WithFallbackPolling enables stat-based detection when watcher init
// or the initial Add fails.
func WithFallbackPolling(d time.Duration) FileOption {
	return func(c *fileConfig) { c.fallbackPolling = d }
}

// WithInitialCallback fires onChange once after a successful watch is
// established.
func WithInitialCallback() FileOption {
	return func(c *fileConfig) { c.initialCallback = true }
}

// WithErrorHandler installs an error handler invoked on watcher
// errors and init failures.
func WithErrorHandler(h func(error)) FileOption {
	return func(c *fileConfig) { c.errorHandler = h }
}

// WatchFile watches path and invokes onChange whenever the file's
// lstat info (size, mode, mtime, inode, or symlink target) changes.
// Blocks until ctx is canceled.
//
// Watch the parent directory so atomic rename updates are observed.
func WatchFile(ctx context.Context, path string, onChange func(), opts ...FileOption) error {
	cfg := fileConfig{}
	for _, o := range opts {
		o(&cfg)
	}
	if onChange == nil {
		onChange = func() {}
	}

	lastSnap := lstatSnapshot(path)

	changed := func() {
		s := lstatSnapshot(path)
		if !s.equal(lastSnap) {
			lastSnap = s
			onChange()
		}
	}

	parent := filepath.Dir(path)

	w, werr := NewWatcher()
	var (
		eventsCh    <-chan Event
		errorsCh    <-chan error
		watchActive bool
	)
	if werr != nil {
		if cfg.errorHandler != nil {
			cfg.errorHandler(werr)
		}
		if cfg.fallbackPolling <= 0 {
			return werr
		}
	} else {
		defer w.Close()
		if err := w.Add(parent); err != nil {
			if cfg.errorHandler != nil {
				cfg.errorHandler(err)
			}
			if cfg.fallbackPolling <= 0 {
				return err
			}
		} else {
			watchActive = true
			eventsCh = w.Events()
			errorsCh = w.Errors()
		}
	}

	// Fire the initial callback only after the watch (or fallback
	// polling) is in place, so a change between the callback and the
	// first event source coming online cannot be missed.
	if cfg.initialCallback {
		onChange()
		lastSnap = lstatSnapshot(path)
	}

	var (
		recheckCh <-chan time.Time
		pollCh    <-chan time.Time
	)
	if cfg.recheckInterval > 0 {
		t := time.NewTicker(cfg.recheckInterval)
		defer t.Stop()
		recheckCh = t.C
	}
	if !watchActive && cfg.fallbackPolling > 0 {
		t := time.NewTicker(cfg.fallbackPolling)
		defer t.Stop()
		pollCh = t.C
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-recheckCh:
			// WatchUntil-style tick: fire onChange unconditionally so
			// callers can implement their own retry semantics.
			onChange()
			lastSnap = lstatSnapshot(path)
		case <-pollCh:
			changed()
		case _, ok := <-eventsCh:
			if !ok {
				eventsCh = nil
				continue
			}
			changed()
		case err, ok := <-errorsCh:
			if !ok {
				errorsCh = nil
				continue
			}
			if errors.Is(err, ErrEventOverflow) {
				// Treat overflow as "treat path as changed."
				changed()
				continue
			}
			if cfg.errorHandler != nil {
				cfg.errorHandler(err)
			}
		}
	}
}

// fileSnapshot captures the parts of os.FileInfo we use for change
// detection plus the resolved symlink target's info if any.
type fileSnapshot struct {
	link   os.FileInfo
	target os.FileInfo
}

func lstatSnapshot(path string) fileSnapshot {
	link, err := os.Lstat(path)
	if err != nil {
		return fileSnapshot{}
	}
	var target os.FileInfo
	if link.Mode()&os.ModeSymlink != 0 {
		target, _ = os.Stat(path)
	}
	return fileSnapshot{link: link, target: target}
}

func (s fileSnapshot) equal(other fileSnapshot) bool {
	return sameInfo(s.link, other.link) && sameInfo(s.target, other.target)
}

func sameInfo(a, b os.FileInfo) bool {
	if (a == nil) != (b == nil) {
		return false
	}
	if a == nil {
		return true
	}
	if a.Size() != b.Size() || a.Mode() != b.Mode() || !a.ModTime().Equal(b.ModTime()) {
		return false
	}
	return os.SameFile(a, b)
}
