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
	"fmt"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// TestClusterDNS checks that cluster DNS works.
func TestClusterDNS(c *client.Client) bool {
	// TODO:
	// https://github.com/GoogleCloudPlatform/kubernetes/issues/3305
	// (but even if it's fixed, this will need a version check for
	// skewed version tests)
	if testContext.provider == "gke" {
		glog.Infof("skipping TestClusterDNS on gke")
		return true
	}

	if testContext.provider == "vagrant" {
		glog.Infof("Skipping test which is broken for vagrant (See https://github.com/GoogleCloudPlatform/kubernetes/issues/3580)")
		return true
	}

	podClient := c.Pods(api.NamespaceDefault)

	//TODO: Wait for skyDNS

	// All the names we need to be able to resolve.
	namesToResolve := []string{
		"kubernetes-ro",
		"kubernetes-ro.default",
		"kubernetes-ro.default.kubernetes.local",
		"google.com",
	}

	probeCmd := "for i in `seq 1 600`; do "
	for _, name := range namesToResolve {
		probeCmd += fmt.Sprintf("wget -O /dev/null %s && echo OK > /results/%s;", name, name)
	}
	probeCmd += "sleep 1; done"

	// Run a pod which probes DNS and exposes the results by HTTP.
	pod := &api.Pod{
		TypeMeta: api.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1beta1",
		},
		ObjectMeta: api.ObjectMeta{
			Name: "dns-test-" + string(util.NewUUID()),
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "results",
					Source: api.VolumeSource{
						EmptyDir: &api.EmptyDir{},
					},
				},
			},
			Containers: []api.Container{
				{
					Name:  "webserver",
					Image: "kubernetes/test-webserver",
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "results",
							MountPath: "/results",
						},
					},
				},
				{
					Name:    "pinger",
					Image:   "busybox",
					Command: []string{"sh", "-c", probeCmd},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "results",
							MountPath: "/results",
						},
					},
				},
			},
		},
	}
	_, err := podClient.Create(pod)
	if err != nil {
		glog.Errorf("Failed to create %s pod: %v", pod.Name, err)
		return false
	}
	defer podClient.Delete(pod.Name)

	waitForPodRunning(c, pod.Name)
	pod, err = podClient.Get(pod.Name)
	if err != nil {
		glog.Errorf("Failed to get pod %s: %v", pod.Name, err)
		return false
	}

	// Try to find results for each expected name.
	var failed []string
	for try := 1; try < 100; try++ {
		failed = []string{}
		for _, name := range namesToResolve {
			_, err := c.Get().
				Prefix("proxy").
				Resource("pods").
				Namespace("default").
				Name(pod.Name).
				Suffix("results", name).
				Do().Raw()
			if err != nil {
				failed = append(failed, name)
				glog.V(4).Infof("Lookup using %s for %s failed: %v", pod.Name, name, err)
			}
		}
		if len(failed) == 0 {
			break
		}
		glog.Infof("lookups using %s failed for: %v", pod.Name, failed)
		time.Sleep(10 * time.Second)
	}
	if len(failed) != 0 {
		glog.Errorf("DNS using %s failed for: %v", pod.Name, failed)
		return false
	}

	// TODO: probe from the host, too.

	glog.Infof("DNS probes using %s succeeded", pod.Name)
	return true
}

var _ = Describe("TestClusterDNS", func() {
	It("should pass", func() {
		// TODO: Instead of OrDie, client should Fail the test if there's a problem.
		// In general tests should Fail() instead of glog.Fatalf().
		Expect(TestClusterDNS(loadClientOrDie())).To(BeTrue())
	})
})
