/*
Copyright 2016 The Kubernetes Authors.

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

// Helper code to initialize the clients used by controllers.
package client

import (
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
)

// Instantiates a new dynamic client pool with the given config and user agent.
func NewDynamicClientPool(cfg *restclient.Config) dynamic.ClientPool {
	// TODO: should use a dynamic RESTMapper built from the discovery results.
	restMapper := registered.RESTMapper()
	return dynamic.NewClientPool(cfg, restMapper, dynamic.LegacyAPIPathResolverFunc)
}
