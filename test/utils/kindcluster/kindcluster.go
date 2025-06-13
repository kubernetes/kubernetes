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
	"os"
	"os/exec"
	"path"
	"regexp"
	"strings"
	"sync"

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

func runKind(tCtx ktesting.TContext, args ...string) {
	tCtx.Helper()
	// In practice, parameters contain no spaces, so no quoting is needed and
	// we can simply concatenate with spaces as separator.
	tCtx.Logf("Running command: kind %s", strings.Join(args, " "))
	cmd := exec.CommandContext(tCtx, "kind", args...)
	var output strings.Builder
	reader, writer := io.Pipe()
	cmd.Stdout = writer
	cmd.Stderr = writer
	tCtx.ExpectNoError(cmd.Start(), "start kind command")
	scanner := bufio.NewScanner(reader)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for scanner.Scan() {
			line := scanner.Text()
			line = strings.TrimSuffix(line, "\n")
			tCtx.Logf("kind: %s", line)
			output.WriteString(line)
			output.WriteByte('\n')
		}
	}()
	tCtx.ExpectNoError(cmd.Wait(), fmt.Sprintf("kind command failed, output:\n%s", output.String()))
	writer.Close()
	wg.Wait()
	tCtx.ExpectNoError(scanner.Err(), "read kind command output")
}
