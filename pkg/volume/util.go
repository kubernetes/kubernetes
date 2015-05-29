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

	"github.com/golang/glog"
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

func ScrubPodVolumeAndWatchUntilCompletion(pod *api.Pod, client client.Interface) error {

	glog.V(5).Infof("Creating scrubber pod for volume %s\n", pod.Name)
	pod, err := client.Pods(api.NamespaceDefault).Create(pod)
	if err != nil {
		return fmt.Errorf("Unexpected error creating a pod to scrub volume %s:  %+v\n", pod.Name, err)
	}

	// the binder will eventually catch up and set status on Claims
	watch := newPodWatch(client, pod.Namespace, pod.Name, 5*time.Second)
	numAttempts := 0
	maxAttempts := 20
	defer watch.Stop()

	success := false
	for {
		event := <-watch.ResultChan()
		pod := event.Object.(*api.Pod)

		glog.V(5).Infof("Handling %s event for pod %+v, current phase %s\n", event.Type, pod.Name, pod.Status.Phase)

		if pod.Status.Phase == api.PodSucceeded {
			glog.V(5).Infof("Success!  Scrub pod %s is phase %s\n", pod.Name, pod.Status.Phase)
			success = true
			break
		} else if pod.Status.Phase == api.PodFailed {
			// success still false
			break
		} else {
			numAttempts++
			glog.V(5).Infof("Pod event %+v\n", pod.Name+" "+string(pod.Status.Phase))
			if numAttempts > maxAttempts {
				break
			}
		}
	}

	err = client.Pods(api.NamespaceDefault).Delete(pod.Name, nil)
	if err != nil {
		return fmt.Errorf("Error deleting scrubber pod: %s\n", pod.Name)
	}

	if success {
		glog.V(5).Infof("Successfully scrubbed volume with pod %s\n", pod.Name)
		return nil
	} else {
		return fmt.Errorf("Volume was not recycled: %+v", pod.Name)
	}
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
