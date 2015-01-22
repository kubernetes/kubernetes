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
	"io/ioutil"
	"path/filepath"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/golang/glog"
)

type testContextType struct {
	authConfig string
	certDir    string
	host       string
	repoRoot   string
	provider   string
}

var testContext testContextType

func waitForPodRunning(c *client.Client, id string) {
	for {
		time.Sleep(5 * time.Second)
		pod, err := c.Pods(api.NamespaceDefault).Get(id)
		if err != nil {
			glog.Warningf("Get pod %s failed: %v", id, err)
			continue
		}
		if pod.Status.Phase == api.PodRunning {
			break
		}
		glog.Infof("Waiting for pod %s status to be %q (found %q)", id, api.PodRunning, pod.Status.Phase)
	}
}

// waitForPodNotPending returns false if it took too long for the pod to go out of pending state.
func waitForPodNotPending(c *client.Client, podName string) bool {
	for i := 0; i < 10; i++ {
		if i > 0 {
			time.Sleep(5 * time.Second)
		}
		pod, err := c.Pods(api.NamespaceDefault).Get(podName)
		if err != nil {
			glog.Warningf("Get pod %s failed: %v", podName, err)
			continue
		}
		if pod.Status.Phase != api.PodPending {
			glog.Infof("Saw pod %s out of pending state (found %q)", podName, pod.Status.Phase)
			return true
		}
		glog.Infof("Waiting for pod %s status to be !%q (found %q)", podName, api.PodPending, pod.Status.Phase)
	}
	glog.Warningf("Gave up waiting for pod %s status to go out of pending", podName)
	return false
}

// waitForPodSuccess returns true if the pod reached state success, or false if it reached failure or ran too long.
func waitForPodSuccess(c *client.Client, podName string, contName string) bool {
	for i := 0; i < 10; i++ {
		if i > 0 {
			time.Sleep(5 * time.Second)
		}
		pod, err := c.Pods(api.NamespaceDefault).Get(podName)
		if err != nil {
			glog.Warningf("Get pod failed: %v", err)
			continue
		}
		// Cannot use pod.Status.Phase == api.PodSucceeded/api.PodFailed due to #2632
		ci, ok := pod.Status.Info[contName]
		if !ok {
			glog.Infof("No Status.Info for container %s in pod %s yet", contName, podName)
		} else {
			if ci.State.Termination != nil {
				if ci.State.Termination.ExitCode == 0 {
					glog.Infof("Saw pod success")
					return true
				} else {
					glog.Infof("Saw pod failure: %+v", ci.State.Termination)
				}
				glog.Infof("Waiting for pod %q status to be success or failure", podName)
			} else {
				glog.Infof("Nil State.Termination for container %s in pod %s so far", contName, podName)
			}
		}
	}
	glog.Warningf("Gave up waiting for pod %q status to be success or failure", podName)
	return false
}

// assetPath returns a path to the requested file; safe on all
// OSes. NOTE: If you use an asset in this test, you MUST add it to
// the KUBE_TEST_PORTABLE array in hack/lib/golang.sh.
func assetPath(pathElements ...string) string {
	return filepath.Join(testContext.repoRoot, filepath.Join(pathElements...))
}

func loadObjectOrDie(filePath string) runtime.Object {
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		glog.Fatalf("Failed to read object: %v", err)
	}
	return decodeObjectOrDie(data)
}

func decodeObjectOrDie(data []byte) runtime.Object {
	obj, err := latest.Codec.Decode(data)
	if err != nil {
		glog.Fatalf("Failed to decode object: %v", err)
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
		Host: testContext.host,
	}
	info, err := clientauth.LoadFromFile(testContext.authConfig)
	if err != nil {
		glog.Fatalf("Error loading auth: %v", err)
	}
	// If the certificate directory is provided, set the cert paths to be there.
	if testContext.certDir != "" {
		glog.Infof("Expecting certs in %v.", testContext.certDir)
		info.CAFile = filepath.Join(testContext.certDir, "ca.crt")
		info.CertFile = filepath.Join(testContext.certDir, "kubecfg.crt")
		info.KeyFile = filepath.Join(testContext.certDir, "kubecfg.key")
	}
	config, err = info.MergeWithConfig(config)
	if err != nil {
		glog.Fatalf("Error creating client")
	}
	c, err := client.New(&config)
	if err != nil {
		glog.Fatalf("Error creating client")
	}
	return c
}

func parsePodOrDie(json string) *api.Pod {
	obj := decodeObjectOrDie([]byte(json))
	pod, ok := obj.(*api.Pod)
	if !ok {
		glog.Fatalf("Failed to cast pod: %v", obj)
	}
	return pod
}

func parseServiceOrDie(json string) *api.Service {
	obj := decodeObjectOrDie([]byte(json))
	service, ok := obj.(*api.Service)
	if !ok {
		glog.Fatalf("Failed to cast service: %v", obj)
	}
	return service
}
