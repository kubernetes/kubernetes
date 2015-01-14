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

package kubelet

import (
	"fmt"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/golang/glog"
)

const ConfigSourceAnnotationKey = "kubernetes.io/config.source"

// PodOperation defines what changes will be made on a pod configuration.
type PodOperation int

const (
	// This is the current pod configuration
	SET PodOperation = iota
	// Pods with the given ids are new to this source
	ADD
	// Pods with the given ids have been removed from this source
	REMOVE
	// Pods with the given ids have been updated in this source
	UPDATE

	// These constants identify the sources of pods
	// Updates from a file
	FileSource = "file"
	// Updates from etcd
	EtcdSource = "etcd"
	// Updates from querying a web page
	HTTPSource = "http"
	// Updates received to the kubelet server
	ServerSource = "server"
	// Updates from Kubernetes API Server
	ApiserverSource = "api"
	// Updates from all sources
	AllSource = "*"
)

// PodUpdate defines an operation sent on the channel. You can add or remove single services by
// sending an array of size one and Op == ADD|REMOVE (with REMOVE, only the ID is required).
// For setting the state of the system to a given state for this source configuration, set
// Pods as desired and Op to SET, which will reset the system state to that specified in this
// operation for this source channel. To remove all pods, set Pods to empty object and Op to SET.
//
// Additionally, Pods should never be nil - it should always point to an empty slice. While
// functionally similar, this helps our unit tests properly check that the correct PodUpdates
// are generated.
type PodUpdate struct {
	Pods   []api.BoundPod
	Op     PodOperation
	Source string
}

// GetPodFullName returns a name that uniquely identifies a pod across all config sources.
func GetPodFullName(pod *api.BoundPod) string {
	return fmt.Sprintf("%s.%s.%s", pod.Name, pod.Namespace, pod.Annotations[ConfigSourceAnnotationKey])
}

// ParsePodFullName unpacks a pod full name and returns the pod name, namespace, and annotations.
// If the pod full name is invalid, empty strings are returend.
func ParsePodFullName(podFullName string) (podName, podNamespace string, podAnnotations map[string]string) {
	parts := strings.Split(podFullName, ".")
	expectedNumFields := 3
	actualNumFields := len(parts)
	if actualNumFields != expectedNumFields {
		glog.Errorf("found a podFullName (%q) with too few fields: expected %d, actual %d.", podFullName, expectedNumFields, actualNumFields)
		return
	}
	podName = parts[0]
	podNamespace = parts[1]
	podAnnotations = make(map[string]string)
	podAnnotations[ConfigSourceAnnotationKey] = parts[2]
	return
}
