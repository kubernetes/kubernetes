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
	"io/ioutil"
	"net/http"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
)

const (
	nautilusImage       = "kubernetes/update-demo:nautilus"
	kittenImage         = "kubernetes/update-demo:kitten"
	updateDemoSelector  = "name=update-demo"
	updateDemoContainer = "update-demo"
	kubectlProxyPort    = 8011
)

var _ = Describe("kubectl", func() {

	// Constants.
	var (
		updateDemoRoot = filepath.Join(root, "examples/update-demo")
		nautilusPath   = filepath.Join(updateDemoRoot, "nautilus-rc.yaml")
		kittenPath     = filepath.Join(updateDemoRoot, "kitten-rc.yaml")
	)

	var cmd *exec.Cmd

	BeforeEach(func() {
		cmd = kubectlCmd("proxy", fmt.Sprintf("--port=%d", kubectlProxyPort))
		if err := cmd.Start(); err != nil {
			Failf("Unable to start kubectl proxy: %v", err)
		}
	})

	AfterEach(func() {
		// Kill the proxy
		if cmd.Process != nil {
			Logf("Stopping kubectl proxy (pid %d)", cmd.Process.Pid)
			cmd.Process.Kill()
		}
	})

	It("should create and stop a replication controller", func() {
		defer cleanup(nautilusPath)

		By("creating a replication controller")
		runKubectl("create", "-f", nautilusPath)
		validateController(nautilusImage, 2, 30*time.Second)
	})

	It("should scale a replication controller", func() {
		defer cleanup(nautilusPath)

		By("creating a replication controller")
		runKubectl("create", "-f", nautilusPath)
		validateController(nautilusImage, 2, 30*time.Second)
		By("scaling down the replication controller")
		runKubectl("resize", "rc", "update-demo-nautilus", "--replicas=1")
		validateController(nautilusImage, 1, 30*time.Second)
		By("scaling up the replication controller")
		runKubectl("resize", "rc", "update-demo-nautilus", "--replicas=2")
		validateController(nautilusImage, 2, 30*time.Second)
	})

	It("should do a rolling update of a replication controller", func() {
		// Cleanup all resources in case we fail somewhere in the middle
		defer cleanup(updateDemoRoot)

		By("creating the initial replication controller")
		runKubectl("create", "-f", nautilusPath)
		validateController(nautilusImage, 2, 30*time.Second)
		By("rollingupdate to new replication controller")
		runKubectl("rollingupdate", "update-demo-nautilus", "--update-period=1s", "-f", kittenPath)
		validateController(kittenImage, 2, 30*time.Second)
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

func validateController(image string, replicas int, timeout time.Duration) {

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
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
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

			data, err := getData(podID)
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

func getData(podID string) (*updateDemoData, error) {
	resp, err := http.Get(fmt.Sprintf("http://localhost:%d/api/v1beta1/proxy/pods/%s/data.json", kubectlProxyPort, podID))
	if err != nil || resp.StatusCode != 200 {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	Logf("got data: %s", body)
	var data updateDemoData
	err = json.Unmarshal(body, &data)
	return &data, err
}

func kubectlCmd(args ...string) *exec.Cmd {
	// TODO: use kubectl binary directly instead of shell wrapper
	path := filepath.Join(root, "cluster/kubectl.sh")
	Logf("Running '%v'", path+" "+strings.Join(args, " "))
	return exec.Command(path, args...)
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
	return stdout.String()
}
