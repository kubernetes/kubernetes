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
	"bytes"
	"fmt"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Framework supports common operations used by e2e tests; it will keep a client & a namespace for you.
// Eventual goal is to merge this with integration test framework.
type Framework struct {
	BaseName string

	Namespace                *api.Namespace
	Client                   *client.Client
	NamespaceDeletionTimeout time.Duration
}

// NewFramework makes a new framework and sets up a BeforeEach/AfterEach for
// you (you can write additional before/after each functions).
func NewFramework(baseName string) *Framework {
	f := &Framework{
		BaseName: baseName,
	}

	BeforeEach(f.beforeEach)
	AfterEach(f.afterEach)

	return f
}

// beforeEach gets a client and makes a namespace.
func (f *Framework) beforeEach() {
	By("Creating a kubernetes client")
	c, err := loadClient()
	Expect(err).NotTo(HaveOccurred())

	f.Client = c

	By("Building a namespace api object")
	namespace, err := createTestingNS(f.BaseName, f.Client)
	Expect(err).NotTo(HaveOccurred())

	f.Namespace = namespace

	if testContext.VerifyServiceAccount {
		By("Waiting for a default service account to be provisioned in namespace")
		err = waitForDefaultServiceAccountInNamespace(c, namespace.Name)
		Expect(err).NotTo(HaveOccurred())
	} else {
		Logf("Skipping waiting for service account")
	}
}

// afterEach deletes the namespace, after reading its events.
func (f *Framework) afterEach() {
	// Print events if the test failed.
	if CurrentGinkgoTestDescription().Failed {
		By(fmt.Sprintf("Collecting events from namespace %q.", f.Namespace.Name))
		events, err := f.Client.Events(f.Namespace.Name).List(labels.Everything(), fields.Everything())
		Expect(err).NotTo(HaveOccurred())

		for _, e := range events.Items {
			Logf("event for %v: %v %v: %v", e.InvolvedObject.Name, e.Source, e.Reason, e.Message)
		}
		// Note that we don't wait for any cleanup to propagate, which means
		// that if you delete a bunch of pods right before ending your test,
		// you may or may not see the killing/deletion/cleanup events.

		dumpAllPodInfo(f.Client)
	}

	// Check whether all nodes are ready after the test.
	if err := allNodesReady(f.Client, time.Minute); err != nil {
		Failf("All nodes should be ready after test, %v", err)
	}

	By(fmt.Sprintf("Destroying namespace %q for this suite.", f.Namespace.Name))

	timeout := 5 * time.Minute
	if f.NamespaceDeletionTimeout != 0 {
		timeout = f.NamespaceDeletionTimeout
	}
	if err := deleteNS(f.Client, f.Namespace.Name, timeout); err != nil {
		Failf("Couldn't delete ns %q: %s", f.Namespace.Name, err)
	}
	// Paranoia-- prevent reuse!
	f.Namespace = nil
	f.Client = nil
}

// WaitForPodRunning waits for the pod to run in the namespace.
func (f *Framework) WaitForPodRunning(podName string) error {
	return waitForPodRunningInNamespace(f.Client, podName, f.Namespace.Name)
}

// Runs the given pod and verifies that the output of exact container matches the desired output.
func (f *Framework) TestContainerOutput(scenarioName string, pod *api.Pod, containerIndex int, expectedOutput []string) {
	testContainerOutput(scenarioName, f.Client, pod, containerIndex, expectedOutput, f.Namespace.Name)
}

// Runs the given pod and verifies that the output of exact container matches the desired regexps.
func (f *Framework) TestContainerOutputRegexp(scenarioName string, pod *api.Pod, containerIndex int, expectedOutput []string) {
	testContainerOutputRegexp(scenarioName, f.Client, pod, containerIndex, expectedOutput, f.Namespace.Name)
}

// WaitForAnEndpoint waits for at least one endpoint to become available in the
// service's corresponding endpoints object.
func (f *Framework) WaitForAnEndpoint(serviceName string) error {
	for {
		// TODO: Endpoints client should take a field selector so we
		// don't have to list everything.
		list, err := f.Client.Endpoints(f.Namespace.Name).List(labels.Everything())
		if err != nil {
			return err
		}
		rv := list.ResourceVersion

		isOK := func(e *api.Endpoints) bool {
			return e.Name == serviceName && len(e.Subsets) > 0 && len(e.Subsets[0].Addresses) > 0
		}
		for i := range list.Items {
			if isOK(&list.Items[i]) {
				return nil
			}
		}

		w, err := f.Client.Endpoints(f.Namespace.Name).Watch(
			labels.Everything(),
			fields.Set{"metadata.name": serviceName}.AsSelector(),
			rv,
		)
		if err != nil {
			return err
		}
		defer w.Stop()

		for {
			val, ok := <-w.ResultChan()
			if !ok {
				// reget and re-watch
				break
			}
			if e, ok := val.Object.(*api.Endpoints); ok {
				if isOK(e) {
					return nil
				}
			}
		}
	}
}

// Write a file using kubectl exec echo <contents> > <path> via specified container
// Because of the primitive technique we're using here, we only allow ASCII alphanumeric characters
func (f *Framework) WriteFileViaContainer(podName, containerName string, path string, contents string) error {
	By("writing a file in the container")
	allowedCharacters := "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
	for _, c := range contents {
		if !strings.ContainsRune(allowedCharacters, c) {
			return fmt.Errorf("Unsupported character in string to write: %v", c)
		}
	}
	command := fmt.Sprintf("echo '%s' > '%s'", contents, path)
	stdout, stderr, err := kubectlExec(f.Namespace.Name, podName, containerName, "--", "/bin/sh", "-c", command)
	if err != nil {
		Logf("error running kubectl exec to write file: %v\nstdout=%v\nstderr=%v)", err, string(stdout), string(stderr))
	}
	return err
}

// Read a file using kubectl exec cat <path>
func (f *Framework) ReadFileViaContainer(podName, containerName string, path string) (string, error) {
	By("reading a file in the container")

	stdout, stderr, err := kubectlExec(f.Namespace.Name, podName, containerName, "--", "cat", path)
	if err != nil {
		Logf("error running kubectl exec to read file: %v\nstdout=%v\nstderr=%v)", err, string(stdout), string(stderr))
	}
	return string(stdout), err
}

func kubectlExec(namespace string, podName, containerName string, args ...string) ([]byte, []byte, error) {
	var stdout, stderr bytes.Buffer
	cmdArgs := []string{
		"exec",
		fmt.Sprintf("--namespace=%v", namespace),
		podName,
		fmt.Sprintf("-c=%v", containerName),
	}
	cmdArgs = append(cmdArgs, args...)

	cmd := kubectlCmd(cmdArgs...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	Logf("Running '%s %s'", cmd.Path, strings.Join(cmd.Args, " "))
	err := cmd.Run()
	return stdout.Bytes(), stderr.Bytes(), err
}
