// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package container defines types for sub-container events and also
// defines an interface for container operation handlers.
package raw

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strings"

	"github.com/google/cadvisor/container/common"
	"github.com/google/cadvisor/container/libcontainer"
	"github.com/google/cadvisor/manager/watcher"

	"github.com/golang/glog"
	"golang.org/x/exp/inotify"
)

type rawContainerWatcher struct {
	// Absolute path to the root of the cgroup hierarchies
	cgroupPaths map[string]string

	cgroupSubsystems *libcontainer.CgroupSubsystems

	// Inotify event watcher.
	watcher *common.InotifyWatcher

	// Signal for watcher thread to stop.
	stopWatcher chan error
}

func NewRawContainerWatcher() (watcher.ContainerWatcher, error) {
	cgroupSubsystems, err := libcontainer.GetCgroupSubsystems()
	if err != nil {
		return nil, fmt.Errorf("failed to get cgroup subsystems: %v", err)
	}
	if len(cgroupSubsystems.Mounts) == 0 {
		return nil, fmt.Errorf("failed to find supported cgroup mounts for the raw factory")
	}

	watcher, err := common.NewInotifyWatcher()
	if err != nil {
		return nil, err
	}

	rawWatcher := &rawContainerWatcher{
		cgroupPaths:      common.MakeCgroupPaths(cgroupSubsystems.MountPoints, "/"),
		cgroupSubsystems: &cgroupSubsystems,
		watcher:          watcher,
		stopWatcher:      make(chan error),
	}

	return rawWatcher, nil
}

func (self *rawContainerWatcher) Start(events chan watcher.ContainerEvent) error {
	// Watch this container (all its cgroups) and all subdirectories.
	for _, cgroupPath := range self.cgroupPaths {
		_, err := self.watchDirectory(cgroupPath, "/")
		if err != nil {
			return err
		}
	}

	// Process the events received from the kernel.
	go func() {
		for {
			select {
			case event := <-self.watcher.Event():
				err := self.processEvent(event, events)
				if err != nil {
					glog.Warningf("Error while processing event (%+v): %v", event, err)
				}
			case err := <-self.watcher.Error():
				glog.Warningf("Error while watching %q:", "/", err)
			case <-self.stopWatcher:
				err := self.watcher.Close()
				if err == nil {
					self.stopWatcher <- err
					return
				}
			}
		}
	}()

	return nil
}

func (self *rawContainerWatcher) Stop() error {
	// Rendezvous with the watcher thread.
	self.stopWatcher <- nil
	return <-self.stopWatcher
}

// Watches the specified directory and all subdirectories. Returns whether the path was
// already being watched and an error (if any).
func (self *rawContainerWatcher) watchDirectory(dir string, containerName string) (bool, error) {
	alreadyWatching, err := self.watcher.AddWatch(containerName, dir)
	if err != nil {
		return alreadyWatching, err
	}

	// Remove the watch if further operations failed.
	cleanup := true
	defer func() {
		if cleanup {
			_, err := self.watcher.RemoveWatch(containerName, dir)
			if err != nil {
				glog.Warningf("Failed to remove inotify watch for %q: %v", dir, err)
			}
		}
	}()

	// TODO(vmarmol): We should re-do this once we're done to ensure directories were not added in the meantime.
	// Watch subdirectories as well.
	entries, err := ioutil.ReadDir(dir)
	if err != nil {
		return alreadyWatching, err
	}
	for _, entry := range entries {
		if entry.IsDir() {
			entryPath := path.Join(dir, entry.Name())
			_, err = self.watchDirectory(entryPath, path.Join(containerName, entry.Name()))
			if err != nil {
				glog.Errorf("Failed to watch directory %q: %v", entryPath, err)
				if os.IsNotExist(err) {
					// The directory may have been removed before watching. Try to watch the other
					// subdirectories. (https://github.com/kubernetes/kubernetes/issues/28997)
					continue
				}
				return alreadyWatching, err
			}
		}
	}

	cleanup = false
	return alreadyWatching, nil
}

func (self *rawContainerWatcher) processEvent(event *inotify.Event, events chan watcher.ContainerEvent) error {
	// Convert the inotify event type to a container create or delete.
	var eventType watcher.ContainerEventType
	switch {
	case (event.Mask & inotify.IN_CREATE) > 0:
		eventType = watcher.ContainerAdd
	case (event.Mask & inotify.IN_DELETE) > 0:
		eventType = watcher.ContainerDelete
	case (event.Mask & inotify.IN_MOVED_FROM) > 0:
		eventType = watcher.ContainerDelete
	case (event.Mask & inotify.IN_MOVED_TO) > 0:
		eventType = watcher.ContainerAdd
	default:
		// Ignore other events.
		return nil
	}

	// Derive the container name from the path name.
	var containerName string
	for _, mount := range self.cgroupSubsystems.Mounts {
		mountLocation := path.Clean(mount.Mountpoint) + "/"
		if strings.HasPrefix(event.Name, mountLocation) {
			containerName = event.Name[len(mountLocation)-1:]
			break
		}
	}
	if containerName == "" {
		return fmt.Errorf("unable to detect container from watch event on directory %q", event.Name)
	}

	// Maintain the watch for the new or deleted container.
	switch eventType {
	case watcher.ContainerAdd:
		// New container was created, watch it.
		alreadyWatched, err := self.watchDirectory(event.Name, containerName)
		if err != nil {
			return err
		}

		// Only report container creation once.
		if alreadyWatched {
			return nil
		}
	case watcher.ContainerDelete:
		// Container was deleted, stop watching for it.
		lastWatched, err := self.watcher.RemoveWatch(containerName, event.Name)
		if err != nil {
			return err
		}

		// Only report container deletion once.
		if !lastWatched {
			return nil
		}
	default:
		return fmt.Errorf("unknown event type %v", eventType)
	}

	// Deliver the event.
	events <- watcher.ContainerEvent{
		EventType:   eventType,
		Name:        containerName,
		WatchSource: watcher.Raw,
	}

	return nil
}
