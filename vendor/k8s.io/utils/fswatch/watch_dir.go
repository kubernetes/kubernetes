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
	"time"
)

// DirOption configures WatchDir.
type DirOption func(*dirConfig)

type dirConfig struct {
	recheckInterval time.Duration
	errorHandler    func(error)
}

// WithDirRecheckInterval fires onChange every d regardless of
// filesystem events. Also drives retries of Add(dir) when the watch
// is inactive (initial Add failed or the directory was removed).
func WithDirRecheckInterval(d time.Duration) DirOption {
	return func(c *dirConfig) { c.recheckInterval = d }
}

// WithDirErrorHandler installs an error handler invoked on watcher
// errors and Add failures.
func WithDirErrorHandler(h func(error)) DirOption {
	return func(c *dirConfig) { c.errorHandler = h }
}

// WatchDir watches dir non-recursively and invokes onChange on every
// filesystem event in dir. Blocks until ctx is canceled.
//
// If NewWatcher or Add(dir) fails, WatchDir reports the error via the
// configured error handler and keeps ticking onChange on the recheck
// interval; it retries Add(dir) on each tick while the watch is
// inactive. Self Remove/Rename of dir also marks the watch inactive
// so a recreated directory regains event-driven updates on the next
// successful Add.
//
// A return value reports an unrecoverable startup error only when no
// recheck interval was configured.
func WatchDir(ctx context.Context, dir string, onChange func(), opts ...DirOption) error {
	cfg := dirConfig{}
	for _, o := range opts {
		o(&cfg)
	}
	if onChange == nil {
		onChange = func() {}
	}
	reportErr := func(err error) {
		if cfg.errorHandler != nil {
			cfg.errorHandler(err)
		}
	}

	w, werr := NewWatcher()
	if werr != nil {
		reportErr(werr)
		if cfg.recheckInterval <= 0 {
			return werr
		}
	} else {
		defer w.Close()
	}

	var (
		watchActive bool
		eventsCh    <-chan Event
		errorsCh    <-chan error
	)
	if w != nil {
		if err := w.Add(dir); err != nil {
			reportErr(err)
			if cfg.recheckInterval <= 0 {
				return err
			}
		} else {
			watchActive = true
			eventsCh = w.Events()
			errorsCh = w.Errors()
		}
	}

	var recheckCh <-chan time.Time
	if cfg.recheckInterval > 0 {
		t := time.NewTicker(cfg.recheckInterval)
		defer t.Stop()
		recheckCh = t.C
	}

	// retryAdd attempts to (re)establish the watch when it is
	// inactive (initial Add failed or the directory was removed).
	retryAdd := func() {
		if watchActive || w == nil {
			return
		}
		if err := w.Add(dir); err != nil {
			reportErr(err)
			return
		}
		watchActive = true
		eventsCh = w.Events()
		errorsCh = w.Errors()
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-recheckCh:
			retryAdd()
			onChange()
		case ev, ok := <-eventsCh:
			if !ok {
				eventsCh = nil
				continue
			}
			if ev.Name == dir && (ev.Has(Remove) || ev.Has(Rename)) {
				watchActive = false
			}
			onChange()
		case err, ok := <-errorsCh:
			if !ok {
				errorsCh = nil
				continue
			}
			if errors.Is(err, ErrEventOverflow) {
				onChange()
				continue
			}
			reportErr(err)
		}
	}
}
