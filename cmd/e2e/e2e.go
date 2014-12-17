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
	"path/filepath"
	goruntime "runtime"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

var (
	authConfig = flag.String("auth_config", os.Getenv("HOME")+"/.kubernetes_auth", "Path to the auth info file.")
	host       = flag.String("host", "", "The host to connect to")
	repoRoot   = flag.String("repo_root", "./", "Root directory of kubernetes repository, for finding test files. Default assumes working directory is repository root")
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

// assetPath returns a path to the requested file; safe on all
// OSes. NOTE: If you use an asset in this test, you MUST add it to
// the KUBE_TEST_PORTABLE array in hack/lib/golang.sh.
func assetPath(pathElements ...string) string {
	return filepath.Join(*repoRoot, filepath.Join(pathElements...))
}

func loadObjectOrDie(filePath string) runtime.Object {
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

	pod := loadPodOrDie(assetPath("api", "examples", "pod.json"))
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

// TestImportantURLs validates that URLs that people depend on haven't moved.
// ***IMPORTANT*** Do *not* fix this test just by changing the path.  If you moved a URL
// you can break upstream dependencies.
func TestImportantURLs(c *client.Client) bool {
	tests := []struct {
		path string
	}{
		{path: "/validate"},
		{path: "/healthz"},
		// TODO: test proxy links here
	}
	ok := true
	for _, test := range tests {
		glog.Infof("testing: %s", test.path)
		data, err := c.RESTClient.Get().
			AbsPath(test.path).
			Do().
			Raw()
		if err != nil {
			glog.Errorf("Failed: %v\nBody: %s", err, string(data))
			ok = false
		}
	}
	return ok
}

// TestKubeletSendsEvent checks that kubelets and scheduler send events about pods scheduling and running.
func TestKubeletSendsEvent(c *client.Client) bool {
	provider := os.Getenv("KUBERNETES_PROVIDER")
	if len(provider) > 0 && provider != "gce" {
		glog.Infof("skipping TestKubeletSendsEvent on cloud provider %s", provider)
		return true
	}
	if provider == "" {
		glog.Info("KUBERNETES_PROVIDER is unset assuming \"gce\"")
	}

	podClient := c.Pods(api.NamespaceDefault)

	pod := loadPodOrDie(assetPath("cmd", "e2e", "pod.json"))
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

	podWithUid, err := podClient.Get(pod.Name)
	if err != nil {
		glog.Errorf("Failed to get pod: %v", err)
		return false
	}

	// Check for scheduler event about the pod.
	glog.Infof("%+v", podWithUid)
	events, err := c.Events(api.NamespaceDefault).List(
		labels.Everything(),
		labels.Set{
			"involvedObject.kind":      "Pod",
			"involvedObject.uid":       podWithUid.UID,
			"involvedObject.namespace": api.NamespaceDefault,
			"source":                   "scheduler",
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
			"involvedObject.uid":       podWithUid.UID,
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

func TestNetwork(c *client.Client) bool {
	ns := api.NamespaceDefault
	svc, err := c.Services(ns).Create(loadObjectOrDie(assetPath(
		"contrib", "for-tests", "network-tester", "service.json",
	)).(*api.Service))
	if err != nil {
		glog.Errorf("unable to create test service: %v", err)
		return false
	}
	// Clean up service
	defer func() {
		if err = c.Services(ns).Delete(svc.Name); err != nil {
			glog.Errorf("unable to delete svc %v: %v", svc.Name, err)
		}
	}()

	rc, err := c.ReplicationControllers(ns).Create(loadObjectOrDie(assetPath(
		"contrib", "for-tests", "network-tester", "rc.json",
	)).(*api.ReplicationController))
	if err != nil {
		glog.Errorf("unable to create test rc: %v", err)
		return false
	}
	// Clean up rc
	defer func() {
		rc.Spec.Replicas = 0
		rc, err = c.ReplicationControllers(ns).Update(rc)
		if err != nil {
			glog.Errorf("unable to modify replica count for rc %v: %v", rc.Name, err)
			return
		}
		if err = c.ReplicationControllers(ns).Delete(rc.Name); err != nil {
			glog.Errorf("unable to delete rc %v: %v", rc.Name, err)
		}
	}()

	const maxAttempts = 60
	for i := 0; i < maxAttempts; i++ {
		time.Sleep(time.Second)
		body, err := c.Get().Path("proxy").Path("services").Path(svc.Name).Path("status").Do().Raw()
		if err != nil {
			glog.Infof("Attempt %v/%v: service/pod still starting. (error: '%v')", i, maxAttempts, err)
			continue
		}
		switch string(body) {
		case "pass":
			glog.Infof("Passed on attempt %v. Cleaning up.", i)
			return true
		case "running":
			glog.Infof("Attempt %v/%v: test still running", i, maxAttempts)
		case "fail":
			if body, err := c.Get().Path("proxy").Path("services").Path(svc.Name).Path("read").Do().Raw(); err != nil {
				glog.Infof("Failed on attempt %v. Cleaning up. Error reading details: %v", i, err)
			} else {
				glog.Infof("Failed on attempt %v. Cleaning up. Details:\n%v", i, string(body))
			}
			return false
		}
	}

	if body, err := c.Get().Path("proxy").Path("services").Path(svc.Name).Path("read").Do().Raw(); err != nil {
		glog.Infof("Timed out. Cleaning up. Error reading details: %v", err)
	} else {
		glog.Infof("Timed out. Cleaning up. Details:\n%v", string(body))
	}

	return false
}

type TestSpec struct {
	// The test to run
	test func(c *client.Client) bool
	// The human readable name of this test
	name string
	// The id for this test.  It should be constant for the life of the test.
	id int
}

type TestInfo struct {
	passed bool
	spec   TestSpec
}

// Output a summary in the TAP (test anything protocol) format for automated processing.
// See http://testanything.org/ for more info
func outputTAPSummary(infoList []TestInfo) {
	glog.Infof("1..%d", len(infoList))
	for _, info := range infoList {
		if info.passed {
			glog.Infof("ok %d - %s", info.spec.id, info.spec.name)
		} else {
			glog.Infof("not ok %d - %s", info.spec.id, info.spec.name)
		}
	}
}

func main() {
	flag.Parse()
	goruntime.GOMAXPROCS(goruntime.NumCPU())
	util.ReallyCrash = true
	util.InitLogs()
	defer util.FlushLogs()

	go func() {
		defer util.FlushLogs()
		time.Sleep(3 * time.Minute)
		glog.Fatalf("This test has timed out.")
	}()

	c := loadClientOrDie()

	// Define the tests.  Important: for a clean test grid, please keep ids for a test constant.
	tests := []TestSpec{
		{TestKubernetesROService, "TestKubernetesROService", 1},
		{TestKubeletSendsEvent, "TestKubeletSendsEvent", 2},
		{TestImportantURLs, "TestImportantURLs", 3},
		{TestPodUpdate, "TestPodUpdate", 4},
		{TestNetwork, "TestNetwork", 5},
	}

	info := []TestInfo{}
	passed := true
	for _, test := range tests {
		testPassed := test.test(c)
		if !testPassed {
			passed = false
		} // TODO: clean up objects created during a test after the test, so cases
		// are independent.
		info = append(info, TestInfo{testPassed, test})
	}
	outputTAPSummary(info)
	if !passed {
		glog.Fatalf("Tests failed")
	}
}
