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
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"

	. "github.com/onsi/ginkgo"
)

const (
	nautilusImage       = "kubernetes/update-demo:nautilus"
	kittenImage         = "kubernetes/update-demo:kitten"
	updateDemoSelector  = "name=update-demo"
	updateDemoContainer = "update-demo"
	validateTimeout     = 10 * time.Minute // TODO: Make this 30 seconds once #4566 is resolved.
	kubectlProxyPort    = 8011
)

var _ = Describe("kubectl", func() {

	// Constants.
	var (
		updateDemoRoot = filepath.Join(testContext.repoRoot, "examples/update-demo")
		nautilusPath   = filepath.Join(updateDemoRoot, "nautilus-rc.yaml")
		kittenPath     = filepath.Join(updateDemoRoot, "kitten-rc.yaml")
	)

	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
	})

	It("should create and stop a replication controller", func() {
		defer cleanup(nautilusPath)

		By("creating a replication controller")
		runKubectl("create", "-f", nautilusPath)
		validateController(c, nautilusImage, 2)
	})

	It("should scale a replication controller", func() {
		defer cleanup(nautilusPath)

		By("creating a replication controller")
		runKubectl("create", "-f", nautilusPath)
		validateController(c, nautilusImage, 2)
		By("scaling down the replication controller")
		runKubectl("resize", "rc", "update-demo-nautilus", "--replicas=1")
		validateController(c, nautilusImage, 1)
		By("scaling up the replication controller")
		runKubectl("resize", "rc", "update-demo-nautilus", "--replicas=2")
		validateController(c, nautilusImage, 2)
	})

	It("should do a rolling update of a replication controller", func() {
		// Cleanup all resources in case we fail somewhere in the middle
		defer cleanup(updateDemoRoot)

		By("creating the initial replication controller")
		runKubectl("create", "-f", nautilusPath)
		validateController(c, nautilusImage, 2)
		By("rollingupdate to new replication controller")
		runKubectl("rollingupdate", "update-demo-nautilus", "--update-period=1s", "-f", kittenPath)
		validateController(c, kittenImage, 2)
	})

})

func cleanup(filePath string) {
	By("using stop to clean up resources")
	runKubectl("stop", "-f", filePath)

	resources := runKubectl("get", "pods,rc", "-l", updateDemoSelector, "--no-headers")
	if resources != "" {
		Failf("Resources left running after stop:\n%s", resources)
	}
}

func validateController(c *client.Client, image string, replicas int) {

	getPodsTemplate := "--template={{range.items}}{{.id}} {{end}}"

	// NB: kubectl adds the "exists" function to the standard template functions.
	// This lets us check to see if the "running" entry exists for each of the containers
	// we care about. Exists will never return an error and it's safe to check a chain of
	// things, any one of which may not exist. In the below template, all of info,
	// containername, and running might be nil, so the normal index function isn't very
	// helpful.
	// This template is unit-tested in kubectl, so if you change it, update the unit test.
	//
	// You can read about the syntax here: http://golang.org/pkg/text/template/
	getContainerStateTemplate := fmt.Sprintf(`--template={{and (exists . "currentState" "info" "%s" "state" "running")}}`, updateDemoContainer)

	getImageTemplate := fmt.Sprintf(`--template={{(index .currentState.info "%s").image}}`, updateDemoContainer)

	By(fmt.Sprintf("waiting for all containers in %s pods to come up.", updateDemoSelector))
	for start := time.Now(); time.Since(start) < validateTimeout; time.Sleep(5 * time.Second) {
		getPodsOutput := runKubectl("get", "pods", "-o", "template", getPodsTemplate, "-l", updateDemoSelector)
		pods := strings.Fields(getPodsOutput)
		if numPods := len(pods); numPods != replicas {
			By(fmt.Sprintf("Replicas for %s: expected=%d actual=%d", updateDemoSelector, replicas, numPods))
			continue
		}
		var runningPods []string
		for _, podID := range pods {
			running := runKubectl("get", "pods", podID, "-o", "template", getContainerStateTemplate)
			if running == "false" {
				Logf("%s is created but not running", podID)
				continue
			}

			currentImage := runKubectl("get", "pods", podID, "-o", "template", getImageTemplate)
			if currentImage != image {
				Logf("%s is created but running wrong image; expected: %s, actual: %s", podID, image, currentImage)
				continue
			}

			data, err := getData(c, podID)
			if err != nil {
				Logf("%s is running right image but fetching data failed: %v", podID, err)
				continue
			}
			if strings.Contains(data.image, image) {
				Logf("%s is running right image but fetched data has the wrong info: %s", podID, data)
				continue
			}

			Logf("%s is verified up and running", podID)
			runningPods = append(runningPods, podID)
		}
		if len(runningPods) == replicas {
			return
		}
	}
	Failf("Timed out waiting for %s pods to reach valid state", updateDemoSelector)
}

type updateDemoData struct {
	image string `json:"image"`
}

func getData(c *client.Client, podID string) (*updateDemoData, error) {
	body, err := c.Get().
		Prefix("proxy").
		Resource("pods").
		Name(podID).
		Suffix("data.json").
		Do().
		Raw()
	if err != nil {
		return nil, err
	}
	Logf("got data: %s", body)
	var data updateDemoData
	err = json.Unmarshal(body, &data)
	return &data, err
}

func kubectlCmd(args ...string) *exec.Cmd {
	defaultArgs := []string{"--auth-path=" + testContext.authConfig}
	if testContext.certDir != "" {
		defaultArgs = append(defaultArgs,
			fmt.Sprintf("--certificate-authority=%s", filepath.Join(testContext.certDir, "ca.crt")),
			fmt.Sprintf("--client-certificate=%s", filepath.Join(testContext.certDir, "kubecfg.crt")),
			fmt.Sprintf("--client-key=%s", filepath.Join(testContext.certDir, "kubecfg.key")))
	}
	kubectlArgs := append(defaultArgs, args...)
	// TODO: Remove this once gcloud writes a proper entry in the kubeconfig file.
	if testContext.provider == "gke" {
		kubectlArgs = append(kubectlArgs, "--server="+testContext.host)
	}
	cmd := exec.Command("kubectl", kubectlArgs...)
	Logf("Running '%s %s'", cmd.Path, strings.Join(cmd.Args, " "))
	return cmd
}

func runKubectl(args ...string) string {
	var stdout, stderr bytes.Buffer
	cmd := kubectlCmd(args...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	if err := cmd.Run(); err != nil {
		Failf("Error running %v:\nCommand stdout:\n%v\nstderr:\n%v\n", cmd, cmd.Stdout, cmd.Stderr)
		return ""
	}
	Logf(stdout.String())
	// TODO: trimspace should be unnecessary after switching to use kubectl binary directly
	return strings.TrimSpace(stdout.String())
}
