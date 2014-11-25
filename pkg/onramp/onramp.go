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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/golang/glog"
)

// New creates a new Kubelet for use in main
func NewOnramp(ec tools.EtcdClient) *Onramp {
	return &Onramp{
		etcdClient:            ec,
	}
}

// Kubelet is the main kubelet implementation.
type Onramp struct {
	etcdClient tools.EtcdClient
}


// Run starts the kubelet reacting to config updates
func (onrmp *Onramp) Run() {

	glog.Infof("In Onramp RUN!\n")
	_, err := onrmp.etcdClient.Watch("/v2/keys/registry/pods/default/", 1, false, nil, nil)

	if (err == nil) {
		glog.Infof("Error on etcd watch: %s\n", err)
		return
	}

}
