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
	"path/filepath"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"

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
)

var _ = Describe("kubectl", func() {
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

	Describe("update-demo", func() {
		var (
			updateDemoRoot = filepath.Join(testContext.RepoRoot, "examples/update-demo")
			nautilusPath   = filepath.Join(updateDemoRoot, "nautilus-rc.yaml")
			kittenPath     = filepath.Join(updateDemoRoot, "kitten-rc.yaml")
		)

		It("should create and stop a replication controller", func() {
			defer cleanup(nautilusPath, updateDemoSelector)

			By("creating a replication controller")
			runKubectl("create", "-f", nautilusPath, fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
		})

		It("should scale a replication controller", func() {
			defer cleanup(nautilusPath, updateDemoSelector)

			By("creating a replication controller")
			runKubectl("create", "-f", nautilusPath, fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			By("scaling down the replication controller")
			runKubectl("resize", "rc", "update-demo-nautilus", "--replicas=1", fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 1, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			By("scaling up the replication controller")
			runKubectl("resize", "rc", "update-demo-nautilus", "--replicas=2", fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
		})

		It("should do a rolling update of a replication controller", func() {
			// Cleanup all resources in case we fail somewhere in the middle
			defer cleanup(updateDemoRoot, updateDemoSelector)

			By("creating the initial replication controller")
			runKubectl("create", "-f", nautilusPath, fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			By("rolling-update to new replication controller")
			runKubectl("rolling-update", "update-demo-nautilus", "--update-period=1s", "-f", kittenPath, fmt.Sprintf("--namespace=%v", ns))
			validateController(c, kittenImage, 2, "update-demo", updateDemoSelector, getUDData("kitten.jpg", ns), ns)
		})
	})

	Describe("guestbook", func() {
		var guestbookPath = filepath.Join(testContext.RepoRoot, "examples/guestbook")

		It("should create and stop a working application", func() {
			if !providerIs("gce", "gke") {
				By(fmt.Sprintf("Skipping guestbook, uses createExternalLoadBalancer, a (gce|gke) feature"))
				return
			}

			defer cleanup(guestbookPath, frontendSelector, redisMasterSelector, redisSlaveSelector)

			By("creating all guestbook components")
			runKubectl("create", "-f", guestbookPath, fmt.Sprintf("--namespace=%v", ns))

			By("validating guestbook app")
			validateGuestbookApp(c, ns)
		})
	})

})

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
