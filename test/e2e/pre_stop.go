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

package e2e

import (
	"encoding/json"
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
)

// partially cloned from webserver.go
type State struct {
	Received map[string]int
}

func testPreStop(c *client.Client, ns string) {
	// This is the server that will receive the preStop notification
	podDescr := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "server",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "server",
					Image: "gcr.io/google_containers/nettest:1.6",
					Ports: []api.ContainerPort{{ContainerPort: 8080}},
				},
			},
		},
	}
	By(fmt.Sprintf("Creating server pod %s in namespace %s", podDescr.Name, ns))
	_, err := c.Pods(ns).Create(podDescr)
	expectNoError(err, fmt.Sprintf("creating pod %s", podDescr.Name))

	// At the end of the test, clean up by removing the pod.
	defer func() {
		By("Deleting the server pod")
		c.Pods(ns).Delete(podDescr.Name, nil)
	}()

	By("Waiting for pods to come up.")
	err = waitForPodRunningInNamespace(c, podDescr.Name, ns)
	expectNoError(err, "waiting for server pod to start")

	val := "{\"Source\": \"prestop\"}"

	podOut, err := c.Pods(ns).Get(podDescr.Name)
	expectNoError(err, "getting pod info")

	preStopDescr := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "tester",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    "tester",
					Image:   "gcr.io/google_containers/busybox",
					Command: []string{"sleep", "600"},
					Lifecycle: &api.Lifecycle{
						PreStop: &api.Handler{
							Exec: &api.ExecAction{
								Command: []string{
									"wget", "-O-", "--post-data=" + val, fmt.Sprintf("http://%s:8080/write", podOut.Status.PodIP),
								},
							},
						},
					},
				},
			},
		},
	}

	By(fmt.Sprintf("Creating tester pod %s in namespace %s", podDescr.Name, ns))
	_, err = c.Pods(ns).Create(preStopDescr)
	expectNoError(err, fmt.Sprintf("creating pod %s", preStopDescr.Name))
	deletePreStop := true

	// At the end of the test, clean up by removing the pod.
	defer func() {
		if deletePreStop {
			By("Deleting the tester pod")
			c.Pods(ns).Delete(preStopDescr.Name, nil)
		}
	}()

	err = waitForPodRunningInNamespace(c, preStopDescr.Name, ns)
	expectNoError(err, "waiting for tester pod to start")

	// Delete the pod with the preStop handler.
	By("Deleting pre-stop pod")
	if err := c.Pods(ns).Delete(preStopDescr.Name, nil); err == nil {
		deletePreStop = false
	}
	expectNoError(err, fmt.Sprintf("deleting pod: %s", preStopDescr.Name))

	// Validate that the server received the web poke.
	err = wait.Poll(time.Second*5, time.Second*60, func() (bool, error) {
		if body, err := c.Get().
			Namespace(ns).Prefix("proxy").
			Resource("pods").
			Name(podDescr.Name).
			Suffix("read").
			DoRaw(); err != nil {
			By(fmt.Sprintf("Error validating prestop: %v", err))
		} else {
			Logf("Saw: %s", string(body))
			state := State{}
			err := json.Unmarshal(body, &state)
			if err != nil {
				Logf("Error parsing: %v", err)
				return false, nil
			}
			if state.Received["prestop"] != 0 {
				return true, nil
			}
		}
		return false, nil
	})
	expectNoError(err, "validating pre-stop.")
}

var _ = Describe("PreStop", func() {
	f := NewFramework("prestop")

	It("should call prestop when killing a pod", func() {
		testPreStop(f.Client, f.Namespace.Name)
	})
})
