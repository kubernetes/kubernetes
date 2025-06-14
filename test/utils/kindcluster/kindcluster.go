/*
Copyright 2025 The Kubernetes Authors.

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

// Package kindcluster contains wrapper code around invoking kind
// and managing the resulting cluster.
//
// E2E tests using this must be labeled with feature.KindCommand.
package kindcluster

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/gomega"
	gtypes "github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func New(tCtx ktesting.TContext) *Cluster {
	c := &Cluster{}
	tCtx.CleanupCtx(c.Stop)
	return c
}

// Cluster represents one kind cluster. Each instance is
// independent from all other instances as long the name
// chosen by the caller is unique on the node.
type Cluster struct {
	running    bool
	name       string
	dir        string
	kubeConfig string
}

// Start brings up the cluster anew. If it was already running, it will be stopped first.

// Will be stopped automatically at the end of the test.
// If the ARTIFACTS env variable is set and the test failed,
// log files of the kind cluster get dumped into
// $ARTIFACTS/<test name>/kind/<cluster name> before stopping it.
//
// The name should be unique to enable running clusters in parallel.
// The config has to be a complete YAML according to https://kind.sigs.k8s.io/docs/user/configuration/.
// The image source can be a directory containing Kubernetes source code or
// a download URL like https://dl.k8s.io/ci/v1.33.1-19+f900f017250646/kubernetes-server-linux-amd64.tar.gz.
func (c *Cluster) Start(tCtx ktesting.TContext, name, kindConfig, imageName string) {
	tCtx.Helper()
	if c.running {
		c.Stop(tCtx)
	}
	c.name = name
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		// Intentional additional lambda function for source code location in log output.
		c.Stop(tCtx)
	})
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		artifacts := os.ExpandEnv("${ARTIFACTS}")
		if c.name == "" || artifacts == "" || !tCtx.Failed() {
			return
		}
		// Sanitize the name:
		// - remove E2E [] tags
		// - replace whitespaces and some special characters with hyphens
		testName := tCtx.Name()
		testName = regexp.MustCompile(`\s*\[[^\]]*\]`).ReplaceAllString(testName, "")
		testName = regexp.MustCompile(`[[:space:]/:()\\]+`).ReplaceAllString(testName, "-")
		testName = strings.Trim(testName, "-")
		logDir := path.Join(artifacts, testName, "kind", c.name)
		tCtx.Logf("exporting kind cluster logs for %s to %s after test failure", c.name, logDir)
		runKind(tCtx, "export", "logs", "--name", c.name, logDir)
	})

	c.dir = tCtx.TempDir()
	config := path.Join(c.dir, "kind.yaml")
	tCtx.ExpectNoError(os.WriteFile(config, []byte(kindConfig), 0644), "create kind config file")
	c.kubeConfig = path.Join(c.dir, "kube.config")
	// --retain ensures that log exporting above works in case of a failure.
	runKind(tCtx, "create", "cluster", "--retain", "--name", name, "--config", config, "--kubeconfig", c.kubeConfig, "--image", imageName)
	c.running = true
	tCtx.Logf("cluster is running, use KUBECONFIG=%s to access it", c.kubeConfig)
}

// Stop ensures that the cluster is not running anymore.
func (c *Cluster) Stop(tCtx ktesting.TContext) {
	tCtx.Helper()
	runKind(tCtx, "delete", "cluster", "--name", c.name)
	c.running = false
	c.name = ""
}

// LoadConfig returns the REST config for the running cluster.
func (c *Cluster) LoadConfig(tCtx ktesting.TContext) *restclient.Config {
	tCtx.Helper()
	cfg, err := clientcmd.LoadFromFile(c.kubeConfig)
	tCtx.ExpectNoError(err, "load KubeConfig")
	config, err := clientcmd.NewDefaultClientConfig(*cfg, nil).ClientConfig()
	tCtx.ExpectNoError(err, "construct REST config")
	return config

}

// UpdateAll replaces all system components with the ones provided via
// the kube-apiserver|kube-controller-manager|kube-proxy|kube-scheduler.tar
// release images in the given directory.
//
// This corresponds conceptually to https://gist.github.com/aojea/2c94034f8e86d08842e5916231eb3fe1.
func (c *Cluster) UpdateAll(tCtx ktesting.TContext, dockerTag, releaseImagesDir string, kubeletBinary string) {
	// Ensure that we have a client for the cluster.
	restConfig := c.LoadConfig(tCtx)
	restConfig.UserAgent = fmt.Sprintf("%s -- kindcluster", restclient.DefaultKubernetesUserAgent())
	tCtx = ktesting.WithRESTConfig(tCtx, restConfig)

	// In the order in which they need to be upgraded: apiserver first, rest doesn't matter.
	controlPlaneComponents := []string{"kube-apiserver", "kube-controller-manager", "kube-scheduler"}
	nodes := regexp.MustCompile(`[[:space:]]+`).Split(runKind(tCtx, "get", "nodes", "--name", c.name), -1)
	workerNodes := make([]string, 0, len(nodes)-1)
	controlPlaneNodes := make([]string, 0, 1)
	for _, node := range nodes {
		if strings.Contains(node, "worker") {
			workerNodes = append(workerNodes, node)
		} else if node != "" {
			controlPlaneNodes = append(controlPlaneNodes, node)
		}
	}

	for _, node := range controlPlaneNodes {
		for _, component := range controlPlaneComponents {
			runKind(tCtx, "load", "image-archive", path.Join(releaseImagesDir, component+".tar"), "--name", c.name, "--nodes", node)
			manifestPath := "/etc/kubernetes/manifests/" + component + ".yaml"
			manifest := runCmd(tCtx, "docker", "exec", node, "cat", manifestPath)
			runAndLogCmd(tCtx, "docker", "exec", node, "sed", "-i", "-r", fmt.Sprintf(`s|^(.*image\:.*)\:.*$|\1-amd64\:%s|`, dockerTag), manifestPath)
			modifiedManifest := runCmd(tCtx, "docker", "exec", node, "cat", manifestPath)
			tCtx.Logf("Patched %s on node %s. Before:\n%s\n\nDiff:\n%s", manifestPath, node, manifest, cmp.Diff(manifest, modifiedManifest))
			ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *v1.Pod {
				pod, err := tCtx.Client().CoreV1().Pods("kube-system").Get(tCtx, fmt.Sprintf("%s-%s-control-plane", component, c.name), metav1.GetOptions{})
				tCtx.ExpectNoError(err)
				return pod
			}).WithTimeout(5*time.Minute).Should(useImage(dockerTag), "%s should have restarted with image %s", component, dockerTag)
		}
	}

	for _, node := range workerNodes {
		runAndLogCmd(tCtx, "docker", "cp", kubeletBinary, node+":/usr/bin/kubelet")
		// systemctl restart checks that the service comes up.
		runAndLogCmd(tCtx, "docker", "exec", node, "systemctl", "restart", "kubelet")
	}
}

// useImage asserts that a pod's container runs with a certain image, identified via a substring (for example, the image version).
func useImage(imageSubString string) gtypes.GomegaMatcher {
	return gomega.HaveField("Status.ContainerStatuses", gomega.ConsistOf(
		gomega.And(
			gomega.HaveField("State.Running", gomega.Not(gomega.BeNil())),
			gomega.HaveField("Image", gomega.ContainSubstring(imageSubString)),
		),
	))
}

// BuildImage ensures that there is a kind node image for the given version
// of Kubernetes. It can build from a directory with Kubernetes source code
// or download URL like https://dl.k8s.io/ci/v1.33.1-19+f900f017250646/kubernetes-server-linux-amd64.tar.gz.
// It returns the image name required to start a cluster.
func BuildImage(tCtx ktesting.TContext, version, imageSource string) string {
	tCtx.Helper()
	imageName := "kind/cluster:" + version
	// Comment this out temporarily when developing locally to save some time
	// if the image was already built before.
	runKind(tCtx, "build", "node-image", "--image", imageName, imageSource)
	return imageName
}

// ServerDownloadURL returns the full URL from which kind can create a node image
// for the given major/minor version of Kubernetes.
//
// This considers only proper releases.
func ServerDownloadURL(tCtx ktesting.TContext, major, minor uint) string {
	url := fmt.Sprintf("https://dl.k8s.io/release/stable-%d.%d.txt", major, minor)
	get, err := http.NewRequestWithContext(tCtx, http.MethodGet, url, nil)
	tCtx.ExpectNoError(err, "construct GET for %s", url)
	resp, err := http.DefaultClient.Do(get)
	tCtx.ExpectNoError(err, "get %s", url)
	if resp.StatusCode != 200 {
		tCtx.Fatalf("get %s: %d - %s", url, resp.StatusCode, resp.Status)
	}
	if resp.Body == nil {
		tCtx.Fatalf("empty response for %s", url)
	}
	defer resp.Body.Close()
	version, err := io.ReadAll(resp.Body)
	tCtx.ExpectNoError(err, "read response body for %s", url)
	return fmt.Sprintf("https://dl.k8s.io/release/%s/kubernetes-server-linux-amd64.tar.gz", string(version))
}

func runKind(tCtx ktesting.TContext, args ...string) string {
	tCtx.Helper()
	return runAndLogCmd(tCtx, "kind", args...)
}

func runAndLogCmd(tCtx ktesting.TContext, name string, args ...string) string {
	tCtx.Helper()
	tCtx.Logf("Running command: %s %s", name, strings.Join(args, " "))
	cmd := exec.CommandContext(tCtx, name, args...)
	var output strings.Builder
	reader, writer := io.Pipe()
	cmd.Stdout = writer
	cmd.Stderr = writer
	tCtx.ExpectNoError(cmd.Start(), "start %s command", name)
	scanner := bufio.NewScanner(reader)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for scanner.Scan() {
			line := scanner.Text()
			line = strings.TrimSuffix(line, "\n")
			tCtx.Logf("%s: %s", name, line)
			output.WriteString(line)
			output.WriteByte('\n')
		}
	}()
	tCtx.ExpectNoError(cmd.Wait(), fmt.Sprintf("%s command failed, output:\n%s", name, output.String()))
	writer.Close()
	wg.Wait()
	tCtx.ExpectNoError(scanner.Err(), "read %s command output", name)

	return output.String()
}

func runCmd(tCtx ktesting.TContext, name string, args ...string) string {
	tCtx.Helper()
	tCtx.Logf("Running command: %s %s", name, strings.Join(args, " "))
	cmd := exec.CommandContext(tCtx, name, args...)
	output, err := cmd.CombinedOutput()
	tCtx.ExpectNoError(err, "command %s failed\noutput:\n%s", name, string(output))
	return string(output)
}
