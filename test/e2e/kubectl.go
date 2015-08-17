/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	nautilusImage            = "gcr.io/google_containers/update-demo:nautilus"
	kittenImage              = "gcr.io/google_containers/update-demo:kitten"
	updateDemoSelector       = "name=update-demo"
	updateDemoContainer      = "update-demo"
	frontendSelector         = "name=frontend"
	redisMasterSelector      = "name=redis-master"
	redisSlaveSelector       = "name=redis-slave"
	kubectlProxyPort         = 8011
	guestbookStartupTimeout  = 10 * time.Minute
	guestbookResponseTimeout = 3 * time.Minute
	simplePodSelector        = "name=nginx"
	simplePodName            = "nginx"
	nginxDefaultOutput       = "Welcome to nginx!"
	simplePodPort            = 80
)

var (
	portForwardRegexp = regexp.MustCompile("Forwarding from 127.0.0.1:([0-9]+) -> 80")
)

var _ = Describe("Kubectl client", func() {
	defer GinkgoRecover()
	var c *client.Client
	var ns string
	var testingNs *api.Namespace
	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
		testingNs, err = createTestingNS("kubectl", c)
		ns = testingNs.Name
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		By(fmt.Sprintf("Destroying namespace for this suite %v", ns))
		if err := c.Namespaces().Delete(ns); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	Describe("Update Demo", func() {
		var updateDemoRoot, nautilusPath, kittenPath string
		BeforeEach(func() {
			updateDemoRoot = filepath.Join(testContext.RepoRoot, "docs/user-guide/update-demo")
			nautilusPath = filepath.Join(updateDemoRoot, "nautilus-rc.yaml")
			kittenPath = filepath.Join(updateDemoRoot, "kitten-rc.yaml")
		})

		It("should create and stop a replication controller", func() {
			defer cleanup(nautilusPath, ns, updateDemoSelector)

			By("creating a replication controller")
			runKubectl("create", "-f", nautilusPath, fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
		})

		It("should scale a replication controller", func() {
			defer cleanup(nautilusPath, ns, updateDemoSelector)

			By("creating a replication controller")
			runKubectl("create", "-f", nautilusPath, fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			By("scaling down the replication controller")
			runKubectl("scale", "rc", "update-demo-nautilus", "--replicas=1", fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 1, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			By("scaling up the replication controller")
			runKubectl("scale", "rc", "update-demo-nautilus", "--replicas=2", fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
		})

		It("should do a rolling update of a replication controller", func() {
			By("creating the initial replication controller")
			runKubectl("create", "-f", nautilusPath, fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			By("rolling-update to new replication controller")
			runKubectl("rolling-update", "update-demo-nautilus", "--update-period=1s", "-f", kittenPath, fmt.Sprintf("--namespace=%v", ns))
			validateController(c, kittenImage, 2, "update-demo", updateDemoSelector, getUDData("kitten.jpg", ns), ns)
			// Everything will hopefully be cleaned up when the namespace is deleted.
		})
	})

	Describe("Guestbook application", func() {
		var guestbookPath string

		BeforeEach(func() {
			guestbookPath = filepath.Join(testContext.RepoRoot, "examples/guestbook")

			// requires ExternalLoadBalancer support
			SkipUnlessProviderIs("gce", "gke", "aws")
		})

		It("should create and stop a working application", func() {
			defer cleanup(guestbookPath, ns, frontendSelector, redisMasterSelector, redisSlaveSelector)

			By("creating all guestbook components")
			runKubectl("create", "-f", guestbookPath, fmt.Sprintf("--namespace=%v", ns))

			By("validating guestbook app")
			validateGuestbookApp(c, ns)
		})
	})

	Describe("Simple pod", func() {
		var podPath string

		BeforeEach(func() {
			podPath = filepath.Join(testContext.RepoRoot, "docs/user-guide/pod.yaml")
			By("creating the pod")
			runKubectl("create", "-f", podPath, fmt.Sprintf("--namespace=%v", ns))
			checkPodsRunningReady(c, ns, []string{simplePodName}, podStartTimeout)

		})
		AfterEach(func() {
			cleanup(podPath, ns, simplePodSelector)
		})

		It("should support exec", func() {
			By("executing a command in the container")
			execOutput := runKubectl("exec", fmt.Sprintf("--namespace=%v", ns), simplePodName, "echo", "running", "in", "container")
			expectedExecOutput := "running in container"
			if execOutput != expectedExecOutput {
				Failf("Unexpected kubectl exec output. Wanted '%s', got '%s'", execOutput, expectedExecOutput)
			}
		})
		It("should support port-forward", func() {
			By("forwarding the container port to a local port")
			cmd := kubectlCmd("port-forward", fmt.Sprintf("--namespace=%v", ns), "-p", simplePodName, fmt.Sprintf(":%d", simplePodPort))
			defer func() {
				if err := cmd.Process.Kill(); err != nil {
					Logf("ERROR failed to kill kubectl port-forward command! The process may leak")
				}
			}()
			// This is somewhat ugly but is the only way to retrieve the port that was picked
			// by the port-forward command. We don't want to hard code the port as we have no
			// way of guaranteeing we can pick one that isn't in use, particularly on Jenkins.
			Logf("starting port-forward command and streaming output")
			stdout, stderr, err := startCmdAndStreamOutput(cmd)
			if err != nil {
				Failf("Failed to start port-forward command: %v", err)
			}
			defer stdout.Close()
			defer stderr.Close()

			buf := make([]byte, 128)
			var n int
			Logf("reading from `kubectl port-forward` command's stderr")
			if n, err = stderr.Read(buf); err != nil {
				Failf("Failed to read from kubectl port-forward stderr: %v", err)
			}
			portForwardOutput := string(buf[:n])
			match := portForwardRegexp.FindStringSubmatch(portForwardOutput)
			if len(match) != 2 {
				Failf("Failed to parse kubectl port-forward output: %s", portForwardOutput)
			}
			By("curling local port output")
			localAddr := fmt.Sprintf("http://localhost:%s", match[1])
			body, err := curl(localAddr)
			if err != nil {
				Failf("Failed http.Get of forwarded port (%s): %v", localAddr, err)
			}
			if !strings.Contains(body, nginxDefaultOutput) {
				Failf("Container port output missing expected value. Wanted:'%s', got: %s", nginxDefaultOutput, body)
			}
		})
	})

	Describe("Kubectl label", func() {
		var podPath string
		var nsFlag string
		BeforeEach(func() {
			podPath = filepath.Join(testContext.RepoRoot, "docs/user-guide/pod.yaml")
			By("creating the pod")
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			runKubectl("create", "-f", podPath, nsFlag)
			checkPodsRunningReady(c, ns, []string{simplePodName}, podStartTimeout)
		})
		AfterEach(func() {
			cleanup(podPath, ns, simplePodSelector)
		})

		It("should update the label on a resource", func() {
			labelName := "testing-label"
			labelValue := "testing-label-value"

			By("adding the label " + labelName + " with value " + labelValue + " to a pod")
			runKubectl("label", "pods", simplePodName, labelName+"="+labelValue, nsFlag)
			By("verifying the pod has the label " + labelName + " with the value " + labelValue)
			output := runKubectl("get", "pod", simplePodName, "-L", labelName, nsFlag)
			if !strings.Contains(output, labelValue) {
				Failf("Failed updating label " + labelName + " to the pod " + simplePodName)
			}

			By("removing the label " + labelName + " of a pod")
			runKubectl("label", "pods", simplePodName, labelName+"-", nsFlag)
			By("verifying the pod doesn't have the label " + labelName)
			output = runKubectl("get", "pod", simplePodName, "-L", labelName, nsFlag)
			if strings.Contains(output, labelValue) {
				Failf("Failed removing label " + labelName + " of the pod " + simplePodName)
			}
		})
	})

})

func curl(addr string) (string, error) {
	resp, err := http.Get(addr)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body[:]), nil
}

func validateGuestbookApp(c *client.Client, ns string) {
	Logf("Waiting for frontend to serve content.")
	if !waitForGuestbookResponse(c, "get", "", `{"data": ""}`, guestbookStartupTimeout, ns) {
		Failf("Frontend service did not start serving content in %v seconds.", guestbookStartupTimeout.Seconds())
	}

	Logf("Trying to add a new entry to the guestbook.")
	if !waitForGuestbookResponse(c, "set", "TestEntry", `{"message": "Updated"}`, guestbookResponseTimeout, ns) {
		Failf("Cannot added new entry in %v seconds.", guestbookResponseTimeout.Seconds())
	}

	Logf("Verifying that added entry can be retrieved.")
	if !waitForGuestbookResponse(c, "get", "", `{"data": "TestEntry"}`, guestbookResponseTimeout, ns) {
		Failf("Entry to guestbook wasn't correctly added in %v seconds.", guestbookResponseTimeout.Seconds())
	}
}

// Returns whether received expected response from guestbook on time.
func waitForGuestbookResponse(c *client.Client, cmd, arg, expectedResponse string, timeout time.Duration, ns string) bool {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		res, err := makeRequestToGuestbook(c, cmd, arg, ns)
		if err == nil && res == expectedResponse {
			return true
		}
	}
	return false
}

func makeRequestToGuestbook(c *client.Client, cmd, value string, ns string) (string, error) {
	result, err := c.Get().
		Prefix("proxy").
		Namespace(ns).
		Resource("services").
		Name("frontend").
		Suffix("/index.php").
		Param("cmd", cmd).
		Param("key", "messages").
		Param("value", value).
		Do().
		Raw()
	return string(result), err
}

type updateDemoData struct {
	Image string
}

// getUDData creates a validator function based on the input string (i.e. kitten.jpg).
// For example, if you send "kitten.jpg", this function veridies that the image jpg = kitten.jpg
// in the container's json field.
func getUDData(jpgExpected string, ns string) func(*client.Client, string) error {

	// getUDData validates data.json in the update-demo (returns nil if data is ok).
	return func(c *client.Client, podID string) error {
		Logf("validating pod %s", podID)
		body, err := c.Get().
			Prefix("proxy").
			Namespace(ns).
			Resource("pods").
			Name(podID).
			Suffix("data.json").
			Do().
			Raw()
		if err != nil {
			return err
		}
		Logf("got data: %s", body)
		var data updateDemoData
		if err := json.Unmarshal(body, &data); err != nil {
			return err
		}
		Logf("Unmarshalled json jpg/img => %s , expecting %s .", data, jpgExpected)
		if strings.Contains(data.Image, jpgExpected) {
			return nil
		} else {
			return errors.New(fmt.Sprintf("data served up in container is innaccurate, %s didn't contain %s", data, jpgExpected))
		}
	}
}
