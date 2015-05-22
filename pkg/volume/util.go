/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package volume

import (
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func GetAccessModesAsString(modes []api.PersistentVolumeAccessMode) string {
	modesAsString := ""

	if contains(modes, api.ReadWriteOnce) {
		appendAccessMode(&modesAsString, "RWO")
	}
	if contains(modes, api.ReadOnlyMany) {
		appendAccessMode(&modesAsString, "ROX")
	}
	if contains(modes, api.ReadWriteMany) {
		appendAccessMode(&modesAsString, "RWX")
	}

	return modesAsString
}

func appendAccessMode(modes *string, mode string) {
	if *modes != "" {
		*modes += ","
	}
	*modes += mode
}

func contains(modes []api.PersistentVolumeAccessMode, mode api.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

// basicVolumeScrubber is an implementation of volume.Recycler
type basicVolumeScrubber struct {
}

// podWatch provides watch semantics for a pod backed by a poller, since
// events aren't generated for pod status updates.
type podWatch struct {
	result chan watch.Event
	stop   chan bool
}

// newPodWatch makes a new podWatch.
func newPodWatch(c client.Interface, namespace, name string, period time.Duration) *podWatch {
	pods := make(chan watch.Event)
	stop := make(chan bool)
	tick := time.NewTicker(period)
	go func() {
		for {
			select {
			case <-stop:
				return
			case <-tick.C:
				pod, err := c.Pods(namespace).Get(name)
				if err != nil {
					pods <- watch.Event{
						Type: watch.Error,
						Object: &api.Status{
							Status:  "Failure",
							Message: fmt.Sprintf("couldn't get pod %s/%s: %s", namespace, name, err),
						},
					}
					continue
				}
				pods <- watch.Event{
					Type:   watch.Modified,
					Object: pod,
				}
			}
		}
	}()

	return &podWatch{
		result: pods,
		stop:   stop,
	}
}

func (w *podWatch) Stop() {
	w.stop <- true
}

func (w *podWatch) ResultChan() <-chan watch.Event {
	return w.result
}
