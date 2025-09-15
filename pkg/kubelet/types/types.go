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

package types

import (
	"net/http"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/cri-client/pkg/logs"
)

// TODO: Reconcile custom types in kubelet/types and this subpackage

// HTTPDoer encapsulates http.Do functionality
type HTTPDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

// Timestamp wraps around time.Time and offers utilities to format and parse
// the time using RFC3339Nano
type Timestamp struct {
	time time.Time
}

// NewTimestamp returns a Timestamp object using the current time.
func NewTimestamp() *Timestamp {
	return &Timestamp{time.Now()}
}

// ConvertToTimestamp takes a string, parses it using the RFC3339NanoLenient layout,
// and converts it to a Timestamp object.
func ConvertToTimestamp(timeString string) *Timestamp {
	parsed, _ := time.Parse(logs.RFC3339NanoLenient, timeString)
	return &Timestamp{parsed}
}

// Get returns the time as time.Time.
func (t *Timestamp) Get() time.Time {
	return t.time
}

// GetString returns the time in the string format using the RFC3339NanoFixed
// layout.
func (t *Timestamp) GetString() string {
	return t.time.Format(logs.RFC3339NanoFixed)
}

// SortedContainerStatuses is a type to help sort container statuses based on container names.
type SortedContainerStatuses []v1.ContainerStatus

func (s SortedContainerStatuses) Len() int      { return len(s) }
func (s SortedContainerStatuses) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

func (s SortedContainerStatuses) Less(i, j int) bool {
	return s[i].Name < s[j].Name
}

// SortInitContainerStatuses ensures that statuses are in the order that their
// init container appears in the pod spec. The function assumes there are no
// duplicate names in the statuses.
func SortInitContainerStatuses(p *v1.Pod, statuses []v1.ContainerStatus) {
	containers := p.Spec.InitContainers
	current := 0
	for _, container := range containers {
		for j := current; j < len(statuses); j++ {
			if container.Name == statuses[j].Name {
				statuses[current], statuses[j] = statuses[j], statuses[current]
				current++
				break
			}
		}
	}
}

// SortStatusesOfInitContainers returns the statuses of InitContainers of pod p,
// in the order that they appear in its spec.
func SortStatusesOfInitContainers(p *v1.Pod, statusMap map[string]*v1.ContainerStatus) []v1.ContainerStatus {
	containers := p.Spec.InitContainers
	statuses := []v1.ContainerStatus{}
	for _, container := range containers {
		if status, found := statusMap[container.Name]; found {
			statuses = append(statuses, *status)
		}
	}
	return statuses
}

// Reservation represents reserved resources for non-pod components.
type Reservation struct {
	// System represents resources reserved for non-kubernetes components.
	System v1.ResourceList
	// Kubernetes represents resources reserved for kubernetes system components.
	Kubernetes v1.ResourceList
}

// ResolvedPodUID is a pod UID which has been translated/resolved to the representation known to kubelets.
type ResolvedPodUID types.UID

// MirrorPodUID is a pod UID for a mirror pod.
type MirrorPodUID types.UID
