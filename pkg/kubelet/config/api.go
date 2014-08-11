/*
Copyright 2014 Google Inc. All rights reserved.

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

// Reads the pod configuration from the Kuberenetes API
package config

import (
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// BoundPodWatcher is the interface for watching for changes to bound pods on a node. Exposed for
// testability.
type BoundPodWatcher interface {
	WatchNodeBoundPods(name string, resourceVersion string) (watch.Interface, error)
}

type SourceAPI struct {
	node    string
	watcher BoundPodWatcher
	updates chan<- interface{}

	waitDuration      time.Duration
	reconnectDuration time.Duration
}

// NewSourceAPI creates a config source that watches for changes to the bound pods by node API. When errors
// occur connecting to the API server, the config source will wait to reconnect for waitDuration.
func NewSourceAPI(node string, watcher BoundPodWatcher, waitDuration time.Duration, updates chan<- interface{}) *SourceAPI {
	config := &SourceAPI{
		node:    node,
		watcher: watcher,
		updates: updates,

		waitDuration: waitDuration,
		// prevent hot loops if the server starts to misbehave
		reconnectDuration: time.Second * 1,
	}
	glog.Infof("Watching apiserver for bound pods for %s", node)
	go util.Forever(config.run, config.waitDuration)
	return config
}

// run loops forever looking for changes to the node pods
func (s *SourceAPI) run() {
	resourceVersion := ""
	for {
		watcher, err := s.watcher.WatchNodeBoundPods(s.node, resourceVersion)
		if err != nil {
			glog.Errorf("Unable to watch for bound pod changes for %s: %v", s.node, err)
			time.Sleep(s.waitDuration)
			continue
		}
		resourceVersion = readBoundPodChannel(s.updates, watcher.ResultChan())
		time.Sleep(s.reconnectDuration)
	}
}

// readBoundPodChannel reads watch events until closed. Returns the last resource version
// seen.
func readBoundPodChannel(dst chan<- interface{}, src <-chan watch.Event) string {
	resourceVersion := ""
	for {
		select {
		case event, ok := <-src:
			if !ok {
				glog.V(2).Infof("WatchNodeBoundPods channel closed")
				return resourceVersion
			}

			pods := event.Object.(*api.BoundPods)
			resourceVersion = pods.ResourceVersion

			switch event.Type {
			case watch.Added, watch.Modified:
				dst <- kubelet.PodUpdate{Op: kubelet.SET, Pods: pods.Items}

			case watch.Deleted:
				dst <- kubelet.PodUpdate{Op: kubelet.SET}
			}
		}
	}
	return resourceVersion
}
