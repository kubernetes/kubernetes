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

package e2e

import (
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func TestPodUpdate(c *client.Client) bool {
	podClient := c.Pods(api.NamespaceDefault)

	name := "pod-update-" + string(util.NewUUID())
	value := strconv.Itoa(time.Now().Nanosecond())
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"name": "foo",
				"time": value,
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "nginx",
					Image: "dockerfile/nginx",
					Ports: []api.Port{{ContainerPort: 80, HostPort: 8080}},
					LivenessProbe: &api.Probe{
						Handler: api.Handler{
							HTTPGet: &api.HTTPGetAction{
								Path: "/index.html",
								Port: util.NewIntOrStringFromInt(8080),
							},
						},
						InitialDelaySeconds: 30,
					},
				},
			},
		},
	}
	_, err := podClient.Create(pod)
	if err != nil {
		glog.Errorf("Failed to create pod: %v", err)
		return false
	}
	defer podClient.Delete(pod.Name)
	waitForPodRunning(c, pod.Name)
	pods, err := podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
	if len(pods.Items) != 1 {
		glog.Errorf("Failed to find the correct pod")
		return false
	}

	podOut, err := podClient.Get(pod.Name)
	if err != nil {
		glog.Errorf("Failed to get pod: %v", err)
		return false
	}
	value = "time" + value
	pod.Labels["time"] = value
	pod.ResourceVersion = podOut.ResourceVersion
	pod.UID = podOut.UID
	pod, err = podClient.Update(pod)
	if err != nil {
		glog.Errorf("Failed to update pod: %v", err)
		return false
	}
	waitForPodRunning(c, pod.Name)
	pods, err = podClient.List(labels.SelectorFromSet(labels.Set(map[string]string{"time": value})))
	if len(pods.Items) != 1 {
		glog.Errorf("Failed to find the correct pod after update.")
		return false
	}
	glog.Infof("pod update OK")
	return true
}

var _ = Describe("TestPodUpdate", func() {
	It("should pass", func() {
		// TODO: Instead of OrDie, client should Fail the test if there's a problem.
		// In general tests should Fail() instead of glog.Fatalf().
		Expect(TestPodUpdate(loadClientOrDie())).To(BeTrue())
	})
})
