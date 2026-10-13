//go:build appengine || (!darwin && !dragonfly && !freebsd && !openbsd && !linux && !netbsd && !solaris && !windows)

package fsnotify

import "errors"

type other struct {
	Events chan Event
	Errors chan error
}

var defaultBufferSize = 0

func newBackend(ev chan Event, errs chan error) (backend, error) {
	return nil, errors.New("fsnotify not supported on the current platform")
}
func (w *other) Close() error                              { return nil }
func (w *other) WatchList() []string                       { return nil }
func (w *other) Add(name string) error                     { return nil }
func (w *other) AddWith(name string, opts ...addOpt) error { return nil }
func (w *other) Remove(name string) error                  { return nil }
func (w *other) xSupports(op Op) bool                      { return false }
