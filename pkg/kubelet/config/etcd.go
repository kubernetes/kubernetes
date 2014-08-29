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

// Reads the pod configuration from etcd using the Kubernetes etcd schema
package config

import (
	"fmt"
	"path"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
	"gopkg.in/v1/yaml"
)

func EtcdKeyForHost(hostname string) string {
	return path.Join("/", "registry", "hosts", hostname, "kubelet")
}

// TODO(lavalamp): Use a watcher interface instead of the etcd client directly
type SourceEtcd struct {
	key     string
	client  tools.EtcdClient
	updates chan<- interface{}

	interval time.Duration
	timeout  time.Duration
}

// NewSourceEtcd creates a config source that watches and pulls from a key in etcd
func NewSourceEtcd(key string, client tools.EtcdClient, updates chan<- interface{}) *SourceEtcd {
	config := &SourceEtcd{
		key:     key,
		client:  client,
		updates: updates,

		timeout: 1 * time.Minute,
	}
	glog.Infof("Watching etcd for %s", key)
	go util.Forever(config.run, time.Second)
	return config
}

// run loops forever looking for changes to a key in etcd
func (s *SourceEtcd) run() {
	index := uint64(0)
	for {
		nextIndex, err := s.fetchNextState(index)
		if err != nil {
			if !tools.IsEtcdNotFound(err) {
				glog.Errorf("Unable to extract from the response (%s): %%v", s.key, err)
			}
			return
		}
		index = nextIndex
	}
}

// fetchNextState fetches the key (or waits for a change to a key) and then returns
// the nextIndex to read.  It will watch no longer than s.waitDuration and then return
func (s *SourceEtcd) fetchNextState(fromIndex uint64) (nextIndex uint64, err error) {
	var response *etcd.Response

	if fromIndex == 0 {
		response, err = s.client.Get(s.key, true, false)
	} else {
		response, err = s.client.Watch(s.key, fromIndex, false, nil, stopChannel(s.timeout))
		if tools.IsEtcdWatchStoppedByUser(err) {
			return fromIndex, nil
		}
	}
	if err != nil {
		return fromIndex, err
	}

	pods, err := responseToPods(response)
	if err != nil {
		glog.Infof("Response was in error: %#v", response)
		return 0, fmt.Errorf("error parsing response: %#v", err)
	}

	glog.Infof("Got state from etcd: %+v", pods)
	s.updates <- kubelet.PodUpdate{pods, kubelet.SET}

	return response.Node.ModifiedIndex + 1, nil
}

// responseToPods takes an etcd Response object, and turns it into a structured list of containers.
// It returns a list of containers, or an error if one occurs.
func responseToPods(response *etcd.Response) ([]kubelet.Pod, error) {
	pods := []kubelet.Pod{}
	if response.Node == nil || len(response.Node.Value) == 0 {
		return pods, fmt.Errorf("no nodes field: %v", response)
	}

	manifests := api.ContainerManifestList{}
	if err := yaml.Unmarshal([]byte(response.Node.Value), &manifests); err != nil {
		return pods, fmt.Errorf("could not unmarshal manifests: %v", err)
	}

	for i, manifest := range manifests.Items {
		name := manifest.ID
		if name == "" {
			name = fmt.Sprintf("%d", i+1)
		}
		pods = append(pods, kubelet.Pod{Name: name, Manifest: manifest})
	}
	return pods, nil
}

// stopChannel creates a channel that is closed after a duration for use with etcd client API.
// If until is 0, the channel will never close.
func stopChannel(until time.Duration) chan bool {
	stop := make(chan bool)
	if until == 0 {
		return stop
	}
	go func() {
		select {
		case <-time.After(until):
		}
		stop <- true
		close(stop)
	}()
	return stop

}
