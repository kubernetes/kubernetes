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

package util

import extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"

// OverlappingReplicaSets sorts a list of ReplicaSets by creation timestamp, using their names as a tie breaker.
type OverlappingReplicaSets []*extensions.ReplicaSet

func (o OverlappingReplicaSets) Len() int      { return len(o) }
func (o OverlappingReplicaSets) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o OverlappingReplicaSets) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(o[j].CreationTimestamp)
}
