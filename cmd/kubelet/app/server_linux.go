/*
Copyright 2015 The Kubernetes Authors.

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

package app

import (
	"github.com/fsnotify/fsnotify"
	"k8s.io/klog/v2"
)

func watchForLockfileContention(path string, done chan struct{}) error {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		klog.ErrorS(err, "Unable to create watcher for lockfile")
		return err
	}
	if err = watcher.Add(path); err != nil {
		klog.ErrorS(err, "Unable to watch lockfile")
		watcher.Close()
		return err
	}
	go func() {
		defer func() {
			close(done)
			watcher.Close()
		}()
		for {
			select {
			case ev := <-watcher.Events:
				if ev.Op == fsnotify.Remove {
					klog.InfoS("inotify event", "event", ev)
					return
				}
			case err = <-watcher.Errors:
				klog.ErrorS(err, "inotify watcher error")
				return
			}
		}
	}()
	return nil
}
