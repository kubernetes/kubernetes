/*
Copyright 2015 Google Inc. All rights reserved.

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
	"path/filepath"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"

	. "github.com/onsi/ginkgo"
)

const (
	nautilusImage            = "kubernetes/update-demo:nautilus"
	kittenImage              = "kubernetes/update-demo:kitten"
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

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
	})

	Describe("update-demo", func() {
		var (
			updateDemoRoot = filepath.Join(testContext.repoRoot, "examples/update-demo/v1beta1")
			nautilusPath   = filepath.Join(updateDemoRoot, "nautilus-rc.yaml")
			kittenPath     = filepath.Join(updateDemoRoot, "kitten-rc.yaml")
		)

		It("should create and stop a replication controller", func() {
			defer cleanup(nautilusPath, updateDemoSelector)

			By("creating a replication controller")
			runKubectl("create", "-f", nautilusPath)
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData)
		})

		It("should scale a replication controller", func() {
			defer cleanup(nautilusPath, updateDemoSelector)

			By("creating a replication controller")
			runKubectl("create", "-f", nautilusPath)
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData)
			By("scaling down the replication controller")
			runKubectl("resize", "rc", "update-demo-nautilus", "--replicas=1")
			validateController(c, nautilusImage, 1, "update-demo", updateDemoSelector, getUDData)
			By("scaling up the replication controller")
			runKubectl("resize", "rc", "update-demo-nautilus", "--replicas=2")
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData)
		})

		It("should do a rolling update of a replication controller", func() {
			// Cleanup all resources in case we fail somewhere in the middle
			defer cleanup(updateDemoRoot, updateDemoSelector)

			By("creating the initial replication controller")
			runKubectl("create", "-f", nautilusPath)
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData)
			By("rollingupdate to new replication controller")
			runKubectl("rollingupdate", "update-demo-nautilus", "--update-period=1s", "-f", kittenPath)
			validateController(c, kittenImage, 2, "update-demo", updateDemoSelector, getUDData)
		})
	})

	Describe("guestbook", func() {
		var guestbookPath = filepath.Join(testContext.repoRoot, "examples/guestbook")

		It("should create and stop a working application", func() {
			defer cleanup(guestbookPath, frontendSelector, redisMasterSelector, redisSlaveSelector)

			By("creating all guestbook components")
			runKubectl("create", "-f", guestbookPath)

			By("validating guestbook app")
			validateGuestbookApp(c)
		})
	})

})

func validateGuestbookApp(c *client.Client) {
	Logf("Waiting for frontend to serve content.")
	if !waitForGuestbookResponse(c, "get", "", `{"data": ""}`, guestbookStartupTimeout) {
		Failf("Frontend service did not start serving content in %v seconds.", guestbookStartupTimeout.Seconds())
	}

	Logf("Trying to add a new entry to the guestbook.")
	if !waitForGuestbookResponse(c, "set", "TestEntry", `{"message": "Updated"}`, guestbookResponseTimeout) {
		Failf("Cannot added new entry in %v seconds.", guestbookResponseTimeout.Seconds())
	}

	Logf("Verifying that added entry can be retrieved.")
	if !waitForGuestbookResponse(c, "get", "", `{"data": "TestEntry"}`, guestbookResponseTimeout) {
		Failf("Entry to guestbook wasn't correctly added in %v seconds.", guestbookResponseTimeout.Seconds())
	}
}

// Returns whether received expected response from guestbook on time.
func waitForGuestbookResponse(c *client.Client, cmd, arg, expectedResponse string, timeout time.Duration) bool {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		res, err := makeRequestToGuestbook(c, cmd, arg)
		if err == nil && res == expectedResponse {
			return true
		}
	}
	return false
}

func makeRequestToGuestbook(c *client.Client, cmd, value string) (string, error) {
	result, err := c.Get().
		Prefix("proxy").
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
	image string `json:"image"`
}

// getUDData validates data.json in the update-demo (returns nil if data is ok).
func getUDData(c *client.Client, podID string) error {
	body, err := c.Get().
		Prefix("proxy").
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

	if strings.Contains(data.image, "update-demo") {
		return nil
	} else {
		return errors.New("data served up in container is innaccurate")
	}
}
