/*
Copyright 2018 The Kubernetes Authors.

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

package pluginwatcher

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
	"k8s.io/klog"

	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

// Watcher is the plugin watcher
type Watcher struct {
	path                string
	deprecatedPath      string
	fs                  utilfs.Filesystem
	fsWatcher           *fsnotify.Watcher
	stopped             chan struct{}
	desiredStateOfWorld cache.DesiredStateOfWorld
}

// NewWatcher provides a new watcher
// deprecatedSockDir refers to a pre-GA directory that was used by older plugins
// for socket registration. New plugins should not use this directory.
func NewWatcher(sockDir string, deprecatedSockDir string, desiredStateOfWorld cache.DesiredStateOfWorld) *Watcher {
	return &Watcher{
		path:                sockDir,
		deprecatedPath:      deprecatedSockDir,
		fs:                  &utilfs.DefaultFs{},
		desiredStateOfWorld: desiredStateOfWorld,
	}
}

// Start watches for the creation and deletion of plugin sockets at the path
func (w *Watcher) Start(stopCh <-chan struct{}) error {
	klog.V(2).Infof("Plugin Watcher Start at %s", w.path)

	w.stopped = make(chan struct{})

	// Creating the directory to be watched if it doesn't exist yet,
	// and walks through the directory to discover the existing plugins.
	if err := w.init(); err != nil {
		return err
	}

	fsWatcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("failed to start plugin fsWatcher, err: %v", err)
	}
	w.fsWatcher = fsWatcher

	// Traverse plugin dir and add filesystem watchers before starting the plugin processing goroutine.
	if err := w.traversePluginDir(w.path); err != nil {
		klog.Errorf("failed to traverse plugin socket path %q, err: %v", w.path, err)
	}

	// Traverse deprecated plugin dir, if specified.
	if len(w.deprecatedPath) != 0 {
		if err := w.traversePluginDir(w.deprecatedPath); err != nil {
			klog.Errorf("failed to traverse deprecated plugin socket path %q, err: %v", w.deprecatedPath, err)
		}
	}

	go func(fsWatcher *fsnotify.Watcher) {
		defer close(w.stopped)
		for {
			select {
			case event := <-fsWatcher.Events:
				//TODO: Handle errors by taking corrective measures
				if event.Op&fsnotify.Create == fsnotify.Create {
					err := w.handleCreateEvent(event)
					if err != nil {
						klog.Errorf("error %v when handling create event: %s", err, event)
					}
				} else if event.Op&fsnotify.Remove == fsnotify.Remove {
					w.handleDeleteEvent(event)
				}
				continue
			case err := <-fsWatcher.Errors:
				if err != nil {
					klog.Errorf("fsWatcher received error: %v", err)
				}
				continue
			case <-stopCh:
				// In case of plugin watcher being stopped by plugin manager, stop
				// probing the creation/deletion of plugin sockets.
				// Also give all pending go routines a chance to complete
				select {
				case <-w.stopped:
				case <-time.After(11 * time.Second):
					klog.Errorf("timeout on stopping watcher")
				}
				w.fsWatcher.Close()
				return
			}
		}
	}(fsWatcher)

	return nil
}

func (w *Watcher) init() error {
	klog.V(4).Infof("Ensuring Plugin directory at %s ", w.path)

	if err := w.fs.MkdirAll(w.path, 0755); err != nil {
		return fmt.Errorf("error (re-)creating root %s: %v", w.path, err)
	}

	return nil
}

// Walks through the plugin directory discover any existing plugin sockets.
// Ignore all errors except root dir not being walkable
func (w *Watcher) traversePluginDir(dir string) error {
	return w.fs.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			if path == dir {
				return fmt.Errorf("error accessing path: %s error: %v", path, err)
			}

			klog.Errorf("error accessing path: %s error: %v", path, err)
			return nil
		}

		switch mode := info.Mode(); {
		case mode.IsDir():
			if w.containsBlacklistedDir(path) {
				return filepath.SkipDir
			}

			if err := w.fsWatcher.Add(path); err != nil {
				return fmt.Errorf("failed to watch %s, err: %v", path, err)
			}
		case mode&os.ModeSocket != 0:
			event := fsnotify.Event{
				Name: path,
				Op:   fsnotify.Create,
			}
			//TODO: Handle errors by taking corrective measures
			if err := w.handleCreateEvent(event); err != nil {
				klog.Errorf("error %v when handling create event: %s", err, event)
			}
		default:
			klog.V(5).Infof("Ignoring file %s with mode %v", path, mode)
		}

		return nil
	})
}

// Handle filesystem notify event.
// Files names:
// - MUST NOT start with a '.'
func (w *Watcher) handleCreateEvent(event fsnotify.Event) error {
	klog.V(6).Infof("Handling create event: %v", event)

	if w.containsBlacklistedDir(event.Name) {
		return nil
	}

	fi, err := os.Stat(event.Name)
	if err != nil {
		return fmt.Errorf("stat file %s failed: %v", event.Name, err)
	}

	if strings.HasPrefix(fi.Name(), ".") {
		klog.V(5).Infof("Ignoring file (starts with '.'): %s", fi.Name())
		return nil
	}

	if !fi.IsDir() {
		if fi.Mode()&os.ModeSocket == 0 {
			klog.V(5).Infof("Ignoring non socket file %s", fi.Name())
			return nil
		}

		return w.handlePluginRegistration(event.Name)
	}

	return w.traversePluginDir(event.Name)
}

func (w *Watcher) handlePluginRegistration(socketPath string) error {
	//TODO: Implement rate limiting to mitigate any DOS kind of attacks.
	// Update desired state of world list of plugins
	// If the socket path does exist in the desired world cache, there's still
	// a possibility that it has been deleted and recreated again before it is
	// removed from the desired world cache, so we still need to call AddOrUpdatePlugin
	// in this case to update the timestamp
	klog.V(2).Infof("Adding socket path or updating timestamp %s to desired state cache", socketPath)
	err := w.desiredStateOfWorld.AddOrUpdatePlugin(socketPath, w.foundInDeprecatedDir(socketPath))
	if err != nil {
		return fmt.Errorf("error adding socket path %s or updating timestamp to desired state cache: %v", socketPath, err)
	}
	return nil
}

func (w *Watcher) handleDeleteEvent(event fsnotify.Event) {
	klog.V(6).Infof("Handling delete event: %v", event)

	socketPath := event.Name
	klog.V(2).Infof("Removing socket path %s from desired state cache", socketPath)
	w.desiredStateOfWorld.RemovePlugin(socketPath)
}

// While deprecated dir is supported, to add extra protection around #69015
// we will explicitly blacklist kubernetes.io directory.
func (w *Watcher) containsBlacklistedDir(path string) bool {
	return strings.HasPrefix(path, w.deprecatedPath+"/kubernetes.io/") ||
		path == w.deprecatedPath+"/kubernetes.io"
}

func (w *Watcher) foundInDeprecatedDir(socketPath string) bool {
	if len(w.deprecatedPath) != 0 {
		if socketPath == w.deprecatedPath {
			return true
		}

		deprecatedPath := w.deprecatedPath
		if !strings.HasSuffix(deprecatedPath, "/") {
			deprecatedPath = deprecatedPath + "/"
		}
		if strings.HasPrefix(socketPath, deprecatedPath) {
			return true
		}
	}
	return false
}
