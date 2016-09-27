/*
Copyright 2014 The Kubernetes Authors.

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

package master

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apiserver"
)

// dynamicLister is used to list resources for dynamic third party
// apis. It implements the apiserver.APIResourceLister interface
type dynamicLister struct {
	m    *Master
	path string
}

func (d dynamicLister) ListAPIResources() []unversioned.APIResource {
	return d.m.getExistingThirdPartyResources(d.path)
}

var _ apiserver.APIResourceLister = &dynamicLister{}
