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

package main

import (
	"flag"
	"io/ioutil"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

var (
	authConfig = flag.String("auth_config", os.Getenv("HOME")+"/.kubernetes_auth", "Path to the auth info file.")
	host       = flag.String("host", "", "The host to connect to")
)

func waitForPodRunning(c *client.Client, id string) {
	for {
		time.Sleep(5 * time.Second)
		pod, err := c.Pods(api.NamespaceDefault).Get(id)
		if err != nil {
			glog.Warningf("Get pod failed: %v", err)
			continue
		}
		if pod.Status.Phase == api.PodRunning {
			break
		}
		glog.Infof("Waiting for pod status to be running (%s)", pod.Status.Phase)
	}
}

func loadObjectOrDie(filePath string) interface{} {
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		glog.Fatalf("Failed to read pod: %v", err)
	}
	obj, err := latest.Codec.Decode(data)
	if err != nil {
		glog.Fatalf("Failed to decode pod: %v", err)
	}
	return obj
}

func loadPodOrDie(filePath string) *api.Pod {
	obj := loadObjectOrDie(filePath)
	pod, ok := obj.(*api.Pod)
	if !ok {
		glog.Fatalf("Failed to load pod: %v", obj)
	}
	return pod
}

func loadClientOrDie() *client.Client {
	config := client.Config{
		Host: *host,
	}
	auth, err := clientauth.LoadFromFile(*authConfig)
	if err != nil {
		glog.Fatalf("Error loading auth: %v", err)
	}
	config, err = auth.MergeWithConfig(config)
	if err != nil {
		glog.Fatalf("Error creating client")
	}
	c, err := client.New(&config)
	if err != nil {
		glog.Fatalf("Error creating client")
	}
	return c
}

func TestKubernetesROService(c *client.Client) bool {
	svc := api.ServiceList{}
	err := c.Get().
		Namespace("default").
		AbsPath("/api/v1beta1/proxy/services/kubernetes-ro/api/v1beta1/services").
		Do().
		Into(&svc)
	if err != nil {
		glog.Errorf("unexpected error listing services using ro service: %v", err)
		return false
	}
	var foundRW, foundRO bool
	for i := range svc.Items {
		if svc.Items[i].Name == "kubernetes" {
			foundRW = true
		}
		if svc.Items[i].Name == "kubernetes-ro" {
			foundRO = true
		}
	}
	if !foundRW {
		glog.Error("no RW service found")
	}
	if !foundRO {
		glog.Error("no RO service found")
	}
	if !foundRW || !foundRO {
		return false
	}
	return true
}

func TestPodUpdate(c *client.Client) bool {
	podClient := c.Pods(api.NamespaceDefault)

	pod := loadPodOrDie("./api/examples/pod.json")
	value := strconv.Itoa(time.Now().Nanosecond())
	pod.Labels["time"] = value

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

// TestKubeletSendsEvent checks that kubelets and scheduler send events about pods scheduling and running.
func TestKubeletSendsEvent(c *client.Client) bool {
	provider := os.Getenv("KUBERNETES_PROVIDER")
	if provider != "gce" {
		glog.Infof("skipping TestKubeletSendsEvent on cloud provider %s", provider)
		return true
	}
	if provider == "" {
		glog.Info("KUBERNETES_PROVIDER is unset assuming \"gce\"")
	}

	podClient := c.Pods(api.NamespaceDefault)

	pod := loadPodOrDie("./cmd/e2e/pod.json")
	value := strconv.Itoa(time.Now().Nanosecond())
	pod.Labels["time"] = value

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

	_, err = podClient.Get(pod.Name)
	if err != nil {
		glog.Errorf("Failed to get pod: %v", err)
		return false
	}

	// Check for scheduler event about the pod.
	events, err := c.Events(api.NamespaceDefault).List(
		labels.Everything(),
		labels.Set{
			"involvedObject.name":      pod.Name,
			"involvedObject.kind":      "Pod",
			"involvedObject.namespace": api.NamespaceDefault,
			"source":                   "scheduler",
			"time":                     value,
		}.AsSelector(),
	)
	if err != nil {
		glog.Error("Error while listing events:", err)
		return false
	}
	if len(events.Items) == 0 {
		glog.Error("Didn't see any scheduler events even though pod was running.")
		return false
	}
	glog.Info("Saw scheduler event for our pod.")

	// Check for kubelet event about the pod.
	events, err = c.Events(api.NamespaceDefault).List(
		labels.Everything(),
		labels.Set{
			"involvedObject.name":      pod.Name,
			"involvedObject.kind":      "BoundPod",
			"involvedObject.namespace": api.NamespaceDefault,
			"source":                   "kubelet",
		}.AsSelector(),
	)
	if err != nil {
		glog.Error("Error while listing events:", err)
		return false
	}
	if len(events.Items) == 0 {
		glog.Error("Didn't see any kubelet events even though pod was running.")
		return false
	}
	glog.Info("Saw kubelet event for our pod.")
	return true
}

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(runtime.NumCPU())
	util.ReallyCrash = true
	util.InitLogs()
	defer util.FlushLogs()

	go func() {
		defer util.FlushLogs()
		time.Sleep(3 * time.Minute)
		glog.Fatalf("This test has timed out.")
	}()

	c := loadClientOrDie()

	tests := []func(c *client.Client) bool{
		TestKubernetesROService,
		TestKubeletSendsEvent,
		// TODO(brendandburns): fix this test and re-add it: TestPodUpdate,
	}

	passed := true
	for _, test := range tests {
		testPassed := test(c)
		if !testPassed {
			passed = false
		}
		// TODO: clean up objects created during a test after the test, so cases
		// are independent.
	}
	if !passed {
		glog.Fatalf("Tests failed")
	}
}
