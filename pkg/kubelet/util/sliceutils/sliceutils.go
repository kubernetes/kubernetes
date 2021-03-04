/*
Copyright 2015 The Kubernetes Authors.

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

package sliceutils

import (
	"k8s.io/api/core/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// StringInSlice returns true if s is in list
func StringInSlice(s string, list []string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}

	return false
}

// PodsByCreationTime makes an array of pods sortable by their creation
// timestamps in ascending order.
type PodsByCreationTime []*v1.Pod

func (s PodsByCreationTime) Len() int {
	return len(s)
}

func (s PodsByCreationTime) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s PodsByCreationTime) Less(i, j int) bool {
	return s[i].CreationTimestamp.Before(&s[j].CreationTimestamp)
}

// ByImageSize makes an array of images sortable by their size in descending
// order.
type ByImageSize []kubecontainer.Image

func (a ByImageSize) Less(i, j int) bool {
	if a[i].Size == a[j].Size {
		return a[i].ID > a[j].ID
	}
	return a[i].Size > a[j].Size
}
func (a ByImageSize) Len() int      { return len(a) }
func (a ByImageSize) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
