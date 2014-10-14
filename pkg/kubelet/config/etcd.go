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

// Reads the pod configuration from etcd using the Kubernetes etcd schema.
package config

import (
	"errors"
	"fmt"
	"path"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

func EtcdKeyForHost(hostname string) string {
	return path.Join("/", "registry", "hosts", hostname, "kubelet")
}

type SourceEtcd struct {
	key     string
	helper  tools.EtcdHelper
	updates chan<- interface{}
}

// NewSourceEtcd creates a config source that watches and pulls from a key in etcd
func NewSourceEtcd(key string, client tools.EtcdClient, updates chan<- interface{}) *SourceEtcd {
	helper := tools.EtcdHelper{
		client,
		latest.Codec,
		tools.RuntimeVersionAdapter{latest.ResourceVersioner},
	}
	source := &SourceEtcd{
		key:     key,
		helper:  helper,
		updates: updates,
	}
	glog.V(1).Infof("Watching etcd for %s", key)
	go util.Forever(source.run, time.Second)
	return source
}

func (s *SourceEtcd) run() {
	watching := s.helper.Watch(s.key, 0)
	for {
		select {
		case event, ok := <-watching.ResultChan():
			if !ok {
				return
			}
			if event.Type == watch.Error {
				glog.Infof("Watch closed (%#v). Reopening.", event.Object)
				watching.Stop()
				return
			}
			pods, err := eventToPods(event)
			if err != nil {
				glog.Errorf("Failed to parse result from etcd watch: %v", err)
				continue
			}

			glog.V(4).Infof("Received state from etcd watch: %+v", pods)
			s.updates <- kubelet.PodUpdate{pods, kubelet.SET}
		}
	}
}

// eventToPods takes a watch.Event object, and turns it into a structured list of pods.
// It returns a list of containers, or an error if one occurs.
func eventToPods(ev watch.Event) ([]kubelet.Pod, error) {
	pods := []kubelet.Pod{}
	manifests, ok := ev.Object.(*api.ContainerManifestList)
	if !ok {
		return pods, errors.New("unable to parse response as ContainerManifestList")
	}

	for i, manifest := range manifests.Items {
		name := manifest.ID
		if name == "" {
			name = fmt.Sprintf("%d", i+1)
		}
		pods = append(pods, kubelet.Pod{
			Name:     name,
			Manifest: manifest})
	}

	return pods, nil
}

func makeContainerKey(machine string) string {
	return "/registry/hosts/" + machine + "/kubelet"
}
