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
	"math/rand"
	"path/filepath"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type testContextType struct {
	authConfig string
	certDir    string
	host       string
	repoRoot   string
	provider   string
}

var testContext testContextType

func Logf(format string, a ...interface{}) {
	fmt.Fprintf(GinkgoWriter, "INFO: "+format+"\n", a...)
}

func Failf(format string, a ...interface{}) {
	Fail(fmt.Sprintf(format, a...), 1)
}

func waitForPodRunning(c *client.Client, id string, tryFor time.Duration) error {
	trySecs := int(tryFor.Seconds())
	for i := 0; i <= trySecs; i += 5 {
		time.Sleep(5 * time.Second)
		pod, err := c.Pods(api.NamespaceDefault).Get(id)
		if err != nil {
			return fmt.Errorf("Get pod %s failed: %v", id, err.Error())
		}
		if pod.Status.Phase == api.PodRunning {
			return nil
		}
		Logf("Waiting for pod %s status to be %q (found %q) (%d secs)", id, api.PodRunning, pod.Status.Phase, i)
	}
	return fmt.Errorf("Gave up waiting for pod %s to be running after %d seconds", id, trySecs)
}

// waitForPodNotPending returns false if it took too long for the pod to go out of pending state.
func waitForPodNotPending(c *client.Client, ns, podName string, tryFor time.Duration) error {
	trySecs := int(tryFor.Seconds())
	for i := 0; i <= trySecs; i += 5 {
		if i > 0 {
			time.Sleep(5 * time.Second)
		}
		pod, err := c.Pods(ns).Get(podName)
		if err != nil {
			Logf("Get pod %s in namespace %s failed, ignoring for 5s: %v", podName, ns, err)
			continue
		}
		if pod.Status.Phase != api.PodPending {
			Logf("Saw pod %s in namespace %s out of pending state (found %q)", podName, ns, pod.Status.Phase)
			return nil
		}
		Logf("Waiting for status of pod %s in namespace %s to be !%q (found %q) (%v secs)", podName, ns, api.PodPending, pod.Status.Phase, i)
	}
	return fmt.Errorf("Gave up waiting for status of pod %s in namespace %s to go out of pending after %d seconds", podName, ns, trySecs)
}

// waitForPodSuccess returns true if the pod reached state success, or false if it reached failure or ran too long.
func waitForPodSuccess(c *client.Client, podName string, contName string, tryFor time.Duration) error {
	trySecs := int(tryFor.Seconds())
	for i := 0; i <= trySecs; i += 5 {
		if i > 0 {
			time.Sleep(5 * time.Second)
		}
		pod, err := c.Pods(api.NamespaceDefault).Get(podName)
		if err != nil {
			Logf("Get pod failed, ignoring for 5s: %v", err)
			continue
		}
		// Cannot use pod.Status.Phase == api.PodSucceeded/api.PodFailed due to #2632
		ci, ok := pod.Status.Info[contName]
		if !ok {
			Logf("No Status.Info for container %s in pod %s yet", contName, podName)
		} else {
			if ci.State.Termination != nil {
				if ci.State.Termination.ExitCode == 0 {
					By("Saw pod success")
					return nil
				} else {
					Logf("Saw pod failure: %+v", ci.State.Termination)
				}
				Logf("Waiting for pod %q status to be success or failure", podName)
			} else {
				Logf("Nil State.Termination for container %s in pod %s so far", contName, podName)
			}
		}
	}
	return fmt.Errorf("Gave up waiting for pod %q status to be success or failure after %d seconds", podName, trySecs)
}

func loadClient() (*client.Client, error) {
	config := client.Config{
		Host: testContext.host,
	}
	info, err := clientauth.LoadFromFile(testContext.authConfig)
	if err != nil {
		return nil, fmt.Errorf("Error loading auth: %v", err.Error())
	}
	// If the certificate directory is provided, set the cert paths to be there.
	if testContext.certDir != "" {
		Logf("Expecting certs in %v.", testContext.certDir)
		info.CAFile = filepath.Join(testContext.certDir, "ca.crt")
		info.CertFile = filepath.Join(testContext.certDir, "kubecfg.crt")
		info.KeyFile = filepath.Join(testContext.certDir, "kubecfg.key")
	}
	config, err = info.MergeWithConfig(config)
	if err != nil {
		return nil, fmt.Errorf("Error creating client: %v", err.Error())
	}
	c, err := client.New(&config)
	if err != nil {
		return nil, fmt.Errorf("Error creating client: %v", err.Error())
	}
	return c, nil
}

// TODO: Allow service names to have the same form as names
//       for pods and replication controllers so we don't
//       need to use such a function and can instead
//       use the UUID utilty function.
func randomSuffix() string {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return strconv.Itoa(r.Int() % 10000)
}

func expectNoError(err error, explain ...interface{}) {
	ExpectWithOffset(1, err).NotTo(HaveOccurred(), explain...)
}
