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

package onramp

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/golang/glog"
)

func NewOnramp(ec tools.EtcdClient, ac *client.Client) *Onramp {
	return &Onramp{
		etcdClient: ec,
		kubeClient: ac,
	}
}

type Onramp struct {
	etcdClient tools.EtcdClient
	kubeClient *client.Client
}

func scanPod(onrmp *Onramp) {
	ns := api.NamespaceAll
	pdi := onrmp.kubeClient.Pods(ns)
	if pdi == nil {
		return
	}
	pods, err := pdi.List(labels.Everything())
	if err != nil {
		return
	}
	for m := range pods.Items {
		glog.Infof("pod name %s\n", pods.Items[m])
	}
}

// Run starts the kubelet reacting to config updates
func (onrmp *Onramp) Run() {

	response, err := onrmp.etcdClient.Watch("/registry/pods/default/", 0, true, nil, nil)
	glog.Infof("Done with watch\n")
	if err != nil {
		glog.Infof("Error on etcd watch: %s\n", err)
		return
	}

	if response.Action == "create" {
		scanPod(onrmp)
	}

}
