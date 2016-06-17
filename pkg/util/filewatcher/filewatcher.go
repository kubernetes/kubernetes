package filewatcher

import (
	"github.com/fsnotify/fsnotify"
	"github.com/golang/glog"
)

func CreateFileWatcher(path string) (*fsnotify.Watcher, error) {
	glog.Infof("Adding file watcher on %s", path)
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}
	if err := watcher.Add(path); err != nil {
		return nil, err
	}
	return watcher, nil
}

type EventHandler func(*fsnotify.Watcher, fsnotify.Event)

type ErrorHandler func(*fsnotify.Watcher, error)

func StartFileEventLoop(watcher *fsnotify.Watcher, eventHandler EventHandler, errorHandler ErrorHandler) {
	for {
		select {
		case event := <-watcher.Events:
			eventHandler(watcher, event)
		case err := <-watcher.Errors:
			errorHandler(watcher, err)
			return
		}
	}
}
