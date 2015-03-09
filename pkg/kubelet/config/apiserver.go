/*
Copyright 2015 Google Inc. All rights reserved.

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

// Reads the pod configuration from the Kubernetes apiserver.
package config

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// NewSourceApiserver creates a config source that watches and pulls from the apiserver.
func NewSourceApiserver(client *client.Client, hostname string, updates chan<- interface{}) {
	lw := cache.NewListWatchFromClient(client, "pods", api.NamespaceAll, labels.OneTermEqualSelector(getHostFieldLabel(client.APIVersion()), hostname))
	newSourceApiserverFromLW(lw, updates)
}

// newSourceApiserverFromLW holds creates a config source that watches and pulls from the apiserver.
func newSourceApiserverFromLW(lw cache.ListerWatcher, updates chan<- interface{}) {
	send := func(objs []interface{}) {
		var pods []api.Pod
		for _, o := range objs {
			pod := o.(*api.Pod)
			// Make a dummy self link so that references to this pod will work.
			pod.SelfLink = "/api/v1beta1/pods/" + pod.Name
			pods = append(pods, *pod)
		}
		updates <- kubelet.PodUpdate{pods, kubelet.SET, kubelet.ApiserverSource}
	}
	cache.NewReflector(lw, &api.Pod{}, cache.NewUndeltaStore(send, cache.MetaNamespaceKeyFunc), 0).Run()
}

func getHostFieldLabel(apiVersion string) string {
	switch apiVersion {
	case "v1beta1", "v1beta2":
		return "DesiredState.Host"
	default:
		return "spec.host"
	}
}
