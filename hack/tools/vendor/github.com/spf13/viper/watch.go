//go:build !js
// +build !js

package viper

import "github.com/fsnotify/fsnotify"

type watcher = fsnotify.Watcher

func newWatcher() (*watcher, error) {
	return fsnotify.NewWatcher()
}
