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

// Reads the pod configuration from file or a directory of files.
package config

import (
	"fmt"
	"os"

	"github.com/golang/glog"
	"golang.org/x/exp/inotify"
	"k8s.io/kubernetes/pkg/api"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

func (s *sourceFile) watch() error {
	_, err := os.Stat(s.path)
	if err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		// Emit an update with an empty PodList to allow FileSource to be marked as seen
		s.updates <- kubetypes.PodUpdate{Pods: []*api.Pod{}, Op: kubetypes.SET, Source: kubetypes.FileSource}
		return fmt.Errorf("path does not exist, ignoring")
	}

	w, err := inotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("Unable to create inotify: %v", err)
	}
	defer w.Close()

	err = w.AddWatch(s.path, inotify.IN_DELETE_SELF|inotify.IN_CREATE|inotify.IN_MOVED_TO|inotify.IN_MODIFY|inotify.IN_MOVED_FROM|inotify.IN_DELETE)
	if err != nil {
		return fmt.Errorf("Unable to create inotify for path %q: %v", s.path, err)
	}

	// Reset store with config files already existing when starting
	if err := s.resetStoreFromPath(); err != nil {
		return fmt.Errorf("Unable to read config path %q: %v", s.path, err)
	}

	for {
		select {
		case event := <-w.Event:
			err = s.processEvent(event)
			if err != nil {
				return fmt.Errorf("Error while processing event (%+v): %v", event, err)
			}
		case err = <-w.Error:
			return fmt.Errorf("Error while watching %q: %v", s.path, err)
		}
	}
	return nil
}

func (s *sourceFile) processEvent(e *inotify.Event) error {
	var eventType podEventType
	switch {
	case (e.Mask & inotify.IN_ISDIR) > 0:
		glog.V(1).Infof("Not recursing into config path %q", s.path)
		return nil
	case (e.Mask & inotify.IN_CREATE) > 0:
		eventType = podAdd
	case (e.Mask & inotify.IN_MOVED_TO) > 0:
		eventType = podAdd
	case (e.Mask & inotify.IN_MODIFY) > 0:
		eventType = podModify
	case (e.Mask & inotify.IN_DELETE) > 0:
		eventType = podDelete
	case (e.Mask & inotify.IN_MOVED_FROM) > 0:
		eventType = podDelete
	case (e.Mask & inotify.IN_DELETE_SELF) > 0:
		return fmt.Errorf("The watched path is deleted.")
	default:
		// Ignore rest events
		return nil
	}

	switch eventType {
	case podAdd, podModify:
		if pod, err := s.extractFromFile(e.Name); err != nil {
			glog.V(1).Infof("Can't process config file %q: %v", e.Name, err)
		} else {
			return s.store.Add(pod)
		}
	case podDelete:
		// Reset the store with current pod configs
		return s.resetStoreFromPath()
	}
	return nil
}
