/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package podutil

import (
	"strings"
	"time"

	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
	"k8s.io/kubernetes/pkg/api"
)

func UUIDAnnotator(containerID string, timestamp time.Time) FilterFunc {
	value := containerID + ";" + timestamp.UTC().Format(time.RFC3339)
	return FilterFunc(func(pod *api.Pod) (bool, error) {
		Annotate(&pod.ObjectMeta, map[string]string{
			meta.TimestampedExecutorContainerUUID: value,
		})
		return true, nil
	})
}

func UUIDAnnotation(pod *api.Pod) (containerID string, timestamp time.Time, ok bool) {
	blob, found := pod.Annotations[meta.TimestampedExecutorContainerUUID]
	if !found {
		return // missing annotation
	}
	i := strings.Index(blob, ";")
	if i < 1 {
		return // mising containerID
	}
	if i >= len(blob)-1 {
		return // missing timestamp
	}
	t, err := time.Parse(time.RFC3339, blob[i+1:])
	if err != nil {
		return // bad timestamp!
	}

	containerID = blob[:i]
	timestamp = t
	ok = true
	return
}
