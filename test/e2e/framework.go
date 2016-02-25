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
	"reflect"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_2"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/metrics"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	maxKubectlExecRetries = 5
)

// Framework supports common operations used by e2e tests; it will keep a client & a namespace for you.
// Eventual goal is to merge this with integration test framework.
type Framework struct {
	BaseName string

	Client        *client.Client
	Clientset_1_2 *release_1_2.Clientset

	Namespace                *api.Namespace   // Every test has at least one namespace
	namespacesToDelete       []*api.Namespace // Some tests have more than one.
	NamespaceDeletionTimeout time.Duration

	gatherer *containerResourceGatherer
	// Constraints that passed to a check which is executed after data is gathered to
	// see if 99% of results are within acceptable bounds. It as to be injected in the test,
	// as expectations vary greatly. Constraints are groupped by the container names.
	addonResourceConstraints map[string]resourceConstraint

	logsSizeWaitGroup    sync.WaitGroup
	logsSizeCloseChannel chan bool
	logsSizeVerifier     *LogsSizeVerifier

	// To make sure that this framework cleans up after itself, no matter what,
	// we install a cleanup action before each test and clear it after.  If we
	// should abort, the AfterSuite hook should run all cleanup actions.
	cleanupHandle CleanupActionHandle
}

type TestDataSummary interface {
	PrintHumanReadable() string
	PrintJSON() string
}

// NewFramework makes a new framework and sets up a BeforeEach/AfterEach for
// you (you can write additional before/after each functions).
func NewFramework(baseName string) *Framework {
	f := &Framework{
		BaseName:                 baseName,
		addonResourceConstraints: make(map[string]resourceConstraint),
	}

	BeforeEach(f.beforeEach)
	AfterEach(f.afterEach)

	return f
}

// beforeEach gets a client and makes a namespace.
func (f *Framework) beforeEach() {
	// The fact that we need this feels like a bug in ginkgo.
	// https://github.com/onsi/ginkgo/issues/222
	f.cleanupHandle = AddCleanupAction(f.afterEach)

	By("Creating a kubernetes client")
	c, err := loadClient()
	Expect(err).NotTo(HaveOccurred())

	f.Client = c
	f.Clientset_1_2 = release_1_2.FromUnversionedClient(c)

	By("Building a namespace api object")
	namespace, err := f.CreateNamespace(f.BaseName, map[string]string{
		"e2e-framework": f.BaseName,
	})
	Expect(err).NotTo(HaveOccurred())

	f.Namespace = namespace

	if testContext.VerifyServiceAccount {
		By("Waiting for a default service account to be provisioned in namespace")
		err = waitForDefaultServiceAccountInNamespace(c, namespace.Name)
		Expect(err).NotTo(HaveOccurred())
	} else {
		Logf("Skipping waiting for service account")
	}

	if testContext.GatherKubeSystemResourceUsageData {
		f.gatherer, err = NewResourceUsageGatherer(c)
		if err != nil {
			Logf("Error while creating NewResourceUsageGatherer: %v", err)
		} else {
			go f.gatherer.startGatheringData()
		}
	}

	if testContext.GatherLogsSizes {
		f.logsSizeWaitGroup = sync.WaitGroup{}
		f.logsSizeWaitGroup.Add(1)
		f.logsSizeCloseChannel = make(chan bool)
		f.logsSizeVerifier = NewLogsVerifier(c, f.logsSizeCloseChannel)
		go func() {
			f.logsSizeVerifier.Run()
			f.logsSizeWaitGroup.Done()
		}()
	}
}

// afterEach deletes the namespace, after reading its events.
func (f *Framework) afterEach() {
	RemoveCleanupAction(f.cleanupHandle)

	// DeleteNamespace at the very end in defer, to avoid any
	// expectation failures preventing deleting the namespace.
	defer func() {
		if testContext.DeleteNamespace {
			for _, ns := range f.namespacesToDelete {
				By(fmt.Sprintf("Destroying namespace %q for this suite.", ns.Name))

				timeout := 5 * time.Minute
				if f.NamespaceDeletionTimeout != 0 {
					timeout = f.NamespaceDeletionTimeout
				}
				if err := deleteNS(f.Client, ns.Name, timeout); err != nil {
					if !apierrs.IsNotFound(err) {
						Failf("Couldn't delete ns %q: %s", ns.Name, err)
					} else {
						Logf("Namespace %v was already deleted", ns.Name)
					}
				}
			}
			f.namespacesToDelete = nil
		} else {
			Logf("Found DeleteNamespace=false, skipping namespace deletion!")
		}

		// Paranoia-- prevent reuse!
		f.Namespace = nil
		f.Client = nil
	}()

	// Print events if the test failed.
	if CurrentGinkgoTestDescription().Failed {
		dumpAllNamespaceInfo(f.Client, f.Namespace.Name)
	}

	summaries := make([]TestDataSummary, 0)
	if testContext.GatherKubeSystemResourceUsageData && f.gatherer != nil {
		By("Collecting resource usage data")
		summaries = append(summaries, f.gatherer.stopAndSummarize([]int{90, 99}, f.addonResourceConstraints))
	}

	if testContext.GatherLogsSizes {
		By("Gathering log sizes data")
		close(f.logsSizeCloseChannel)
		f.logsSizeWaitGroup.Wait()
		summaries = append(summaries, f.logsSizeVerifier.GetSummary())
	}

	if testContext.GatherMetricsAfterTest {
		By("Gathering metrics")
		// TODO: enable Scheduler and ControllerManager metrics grabbing when Master's Kubelet will be registered.
		grabber, err := metrics.NewMetricsGrabber(f.Client, true, false, false, true)
		if err != nil {
			Logf("Failed to create MetricsGrabber. Skipping metrics gathering.")
		} else {
			received, err := grabber.Grab(nil)
			if err != nil {
				Logf("MetricsGrabber failed grab metrics. Skipping metrics gathering.")
			} else {
				summaries = append(summaries, (*MetricsForE2E)(&received))
			}
		}
	}

	outputTypes := strings.Split(testContext.OutputPrintType, ",")
	for _, printType := range outputTypes {
		switch printType {
		case "hr":
			for i := range summaries {
				Logf(summaries[i].PrintHumanReadable())
			}
		case "json":
			for i := range summaries {
				typeName := reflect.TypeOf(summaries[i]).String()
				Logf("%v JSON\n%v", typeName[strings.LastIndex(typeName, ".")+1:len(typeName)], summaries[i].PrintJSON())
				Logf("Finished")
			}
		default:
			Logf("Unknown output type: %v. Skipping.", printType)
		}
	}

	// Check whether all nodes are ready after the test.
	// This is explicitly done at the very end of the test, to avoid
	// e.g. not removing namespace in case of this failure.
	if err := allNodesReady(f.Client, time.Minute); err != nil {
		Failf("All nodes should be ready after test, %v", err)
	}
}

func (f *Framework) CreateNamespace(baseName string, labels map[string]string) (*api.Namespace, error) {
	createTestingNS := testContext.CreateTestingNS
	if createTestingNS == nil {
		createTestingNS = CreateTestingNS
	}
	ns, err := createTestingNS(baseName, f.Client, labels)
	if err == nil {
		f.namespacesToDelete = append(f.namespacesToDelete, ns)
	}
	return ns, err
}

// WaitForPodTerminated waits for the pod to be terminated with the given reason.
func (f *Framework) WaitForPodTerminated(podName, reason string) error {
	return waitForPodTerminatedInNamespace(f.Client, podName, reason, f.Namespace.Name)
}

// WaitForPodRunning waits for the pod to run in the namespace.
func (f *Framework) WaitForPodRunning(podName string) error {
	return waitForPodRunningInNamespace(f.Client, podName, f.Namespace.Name)
}

// WaitForPodRunningSlow waits for the pod to run in the namespace.
// It has a longer timeout then WaitForPodRunning (util.slowPodStartTimeout).
func (f *Framework) WaitForPodRunningSlow(podName string) error {
	return waitForPodRunningInNamespaceSlow(f.Client, podName, f.Namespace.Name)
}

// WaitForPodNoLongerRunning waits for the pod to no longer be running in the namespace, for either
// success or failure.
func (f *Framework) WaitForPodNoLongerRunning(podName string) error {
	return waitForPodNoLongerRunningInNamespace(f.Client, podName, f.Namespace.Name)
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
		list, err := f.Client.Endpoints(f.Namespace.Name).List(api.ListOptions{})
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

		options := api.ListOptions{
			FieldSelector:   fields.Set{"metadata.name": serviceName}.AsSelector(),
			ResourceVersion: rv,
		}
		w, err := f.Client.Endpoints(f.Namespace.Name).Watch(options)
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
	stdout, stderr, err := kubectlExecWithRetry(f.Namespace.Name, podName, containerName, "--", "/bin/sh", "-c", command)
	if err != nil {
		Logf("error running kubectl exec to write file: %v\nstdout=%v\nstderr=%v)", err, string(stdout), string(stderr))
	}
	return err
}

// Read a file using kubectl exec cat <path>
func (f *Framework) ReadFileViaContainer(podName, containerName string, path string) (string, error) {
	By("reading a file in the container")

	stdout, stderr, err := kubectlExecWithRetry(f.Namespace.Name, podName, containerName, "--", "cat", path)
	if err != nil {
		Logf("error running kubectl exec to read file: %v\nstdout=%v\nstderr=%v)", err, string(stdout), string(stderr))
	}
	return string(stdout), err
}

func kubectlExecWithRetry(namespace string, podName, containerName string, args ...string) ([]byte, []byte, error) {
	for numRetries := 0; numRetries < maxKubectlExecRetries; numRetries++ {
		if numRetries > 0 {
			Logf("Retrying kubectl exec (retry count=%v/%v)", numRetries+1, maxKubectlExecRetries)
		}

		stdOutBytes, stdErrBytes, err := kubectlExec(namespace, podName, containerName, args...)
		if err != nil {
			if strings.Contains(strings.ToLower(string(stdErrBytes)), "i/o timeout") {
				// Retry on "i/o timeout" errors
				Logf("Warning: kubectl exec encountered i/o timeout.\nerr=%v\nstdout=%v\nstderr=%v)", err, string(stdOutBytes), string(stdErrBytes))
				continue
			}
		}

		return stdOutBytes, stdErrBytes, err
	}
	err := fmt.Errorf("Failed: kubectl exec failed %d times with \"i/o timeout\". Giving up.", maxKubectlExecRetries)
	return nil, nil, err
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
