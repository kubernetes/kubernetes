// +build js,wasm

package viper

import (
	"errors"

	"github.com/fsnotify/fsnotify"
)

type watcher struct {
	Events chan fsnotify.Event
	Errors chan error
}

func (*watcher) Close() error {
	return nil
}

func (*watcher) Add(name string) error {
	return nil
}

func (*watcher) Remove(name string) error {
	return nil
}

func newWatcher() (*watcher, error) {
	return &watcher{}, errors.New("fsnotify is not supported on WASM")
}
