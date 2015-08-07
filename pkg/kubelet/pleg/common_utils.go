/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package pleg

import (
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/sets"
)

// buildIDMap returns a container ID to pod ID map.
func buildIDMap(pods []*kubecontainer.Pod) map[string]types.UID {
	cmap := make(map[string]types.UID)
	for _, p := range pods {
		for _, c := range p.Containers {
			cmap[string(c.ID)] = p.ID
		}
	}
	return cmap
}

// buildContainerSet returns set of container IDs.
func buildContainerSet(pods []*kubecontainer.Pod) sets.String {
	cset := sets.NewString()
	for _, p := range pods {
		for _, c := range p.Containers {
			cset.Insert(string(c.ID))
		}
	}
	return cset
}
