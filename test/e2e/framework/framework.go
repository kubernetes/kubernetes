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

package framework

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
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	adapter_1_2 "k8s.io/kubernetes/pkg/client/unversioned/adapters/release_1_2"
	adapter_1_3 "k8s.io/kubernetes/pkg/client/unversioned/adapters/release_1_3"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/metrics"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"

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
	Clientset_1_3 *release_1_3.Clientset

	Namespace                *api.Namespace   // Every test has at least one namespace
	namespacesToDelete       []*api.Namespace // Some tests have more than one.
	NamespaceDeletionTimeout time.Duration

	gatherer *containerResourceGatherer
	// Constraints that passed to a check which is executed after data is gathered to
	// see if 99% of results are within acceptable bounds. It as to be injected in the test,
	// as expectations vary greatly. Constraints are groupped by the container names.
	AddonResourceConstraints map[string]ResourceConstraint

	logsSizeWaitGroup    sync.WaitGroup
	logsSizeCloseChannel chan bool
	logsSizeVerifier     *LogsSizeVerifier

	// To make sure that this framework cleans up after itself, no matter what,
	// we install a Cleanup action before each test and clear it after.  If we
	// should abort, the AfterSuite hook should run all Cleanup actions.
	cleanupHandle CleanupActionHandle

	// configuration for framework's client
	options FrameworkOptions
}

type TestDataSummary interface {
	PrintHumanReadable() string
	PrintJSON() string
}

type FrameworkOptions struct {
	ClientQPS   float32
	ClientBurst int
}

// NewFramework makes a new framework and sets up a BeforeEach/AfterEach for
// you (you can write additional before/after each functions).
func NewDefaultFramework(baseName string) *Framework {
	options := FrameworkOptions{
		ClientQPS:   20,
		ClientBurst: 50,
	}
	return NewFramework(baseName, options, nil)
}

func NewFramework(baseName string, options FrameworkOptions, client *client.Client) *Framework {
	f := &Framework{
		BaseName:                 baseName,
		AddonResourceConstraints: make(map[string]ResourceConstraint),
		options:                  options,
		Client:                   client,
	}

	BeforeEach(f.BeforeEach)
	AfterEach(f.AfterEach)

	return f
}

// BeforeEach gets a client and makes a namespace.
func (f *Framework) BeforeEach() {
	// The fact that we need this feels like a bug in ginkgo.
	// https://github.com/onsi/ginkgo/issues/222
	f.cleanupHandle = AddCleanupAction(f.AfterEach)

	if f.Client == nil {
		By("Creating a kubernetes client")
		config, err := LoadConfig()
		Expect(err).NotTo(HaveOccurred())
		config.QPS = f.options.ClientQPS
		config.Burst = f.options.ClientBurst
		if TestContext.KubeAPIContentType != "" {
			config.ContentType = TestContext.KubeAPIContentType
		}
		c, err := loadClientFromConfig(config)
		Expect(err).NotTo(HaveOccurred())
		f.Client = c
	}
	f.Clientset_1_2 = adapter_1_2.FromUnversionedClient(f.Client)
	f.Clientset_1_3 = adapter_1_3.FromUnversionedClient(f.Client)

	By("Building a namespace api object")
	namespace, err := f.CreateNamespace(f.BaseName, map[string]string{
		"e2e-framework": f.BaseName,
	})
	Expect(err).NotTo(HaveOccurred())

	f.Namespace = namespace

	if TestContext.VerifyServiceAccount {
		By("Waiting for a default service account to be provisioned in namespace")
		err = WaitForDefaultServiceAccountInNamespace(f.Client, namespace.Name)
		Expect(err).NotTo(HaveOccurred())
	} else {
		Logf("Skipping waiting for service account")
	}

	if TestContext.GatherKubeSystemResourceUsageData {
		f.gatherer, err = NewResourceUsageGatherer(f.Client, ResourceGathererOptions{inKubemark: ProviderIs("kubemark")})
		if err != nil {
			Logf("Error while creating NewResourceUsageGatherer: %v", err)
		} else {
			go f.gatherer.startGatheringData()
		}
	}

	if TestContext.GatherLogsSizes {
		f.logsSizeWaitGroup = sync.WaitGroup{}
		f.logsSizeWaitGroup.Add(1)
		f.logsSizeCloseChannel = make(chan bool)
		f.logsSizeVerifier = NewLogsVerifier(f.Client, f.logsSizeCloseChannel)
		go func() {
			f.logsSizeVerifier.Run()
			f.logsSizeWaitGroup.Done()
		}()
	}
}

// AfterEach deletes the namespace, after reading its events.
func (f *Framework) AfterEach() {
	RemoveCleanupAction(f.cleanupHandle)

	// DeleteNamespace at the very end in defer, to avoid any
	// expectation failures preventing deleting the namespace.
	defer func() {
		if TestContext.DeleteNamespace {
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
		DumpAllNamespaceInfo(f.Client, f.Namespace.Name)
	}

	summaries := make([]TestDataSummary, 0)
	if TestContext.GatherKubeSystemResourceUsageData && f.gatherer != nil {
		By("Collecting resource usage data")
		summaries = append(summaries, f.gatherer.stopAndSummarize([]int{90, 99, 100}, f.AddonResourceConstraints))
	}

	if TestContext.GatherLogsSizes {
		By("Gathering log sizes data")
		close(f.logsSizeCloseChannel)
		f.logsSizeWaitGroup.Wait()
		summaries = append(summaries, f.logsSizeVerifier.GetSummary())
	}

	if TestContext.GatherMetricsAfterTest {
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

	outputTypes := strings.Split(TestContext.OutputPrintType, ",")
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
	if err := AllNodesReady(f.Client, time.Minute); err != nil {
		Failf("All nodes should be ready after test, %v", err)
	}
}

func (f *Framework) CreateNamespace(baseName string, labels map[string]string) (*api.Namespace, error) {
	createTestingNS := TestContext.CreateTestingNS
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
	return WaitForPodRunningInNamespace(f.Client, podName, f.Namespace.Name)
}

// WaitForPodReady waits for the pod to flip to ready in the namespace.
func (f *Framework) WaitForPodReady(podName string) error {
	return waitTimeoutForPodReadyInNamespace(f.Client, podName, f.Namespace.Name, PodStartTimeout)
}

// WaitForPodRunningSlow waits for the pod to run in the namespace.
// It has a longer timeout then WaitForPodRunning (util.slowPodStartTimeout).
func (f *Framework) WaitForPodRunningSlow(podName string) error {
	return waitForPodRunningInNamespaceSlow(f.Client, podName, f.Namespace.Name)
}

// WaitForPodNoLongerRunning waits for the pod to no longer be running in the namespace, for either
// success or failure.
func (f *Framework) WaitForPodNoLongerRunning(podName string) error {
	return WaitForPodNoLongerRunningInNamespace(f.Client, podName, f.Namespace.Name)
}

// Runs the given pod and verifies that the output of exact container matches the desired output.
func (f *Framework) TestContainerOutput(scenarioName string, pod *api.Pod, containerIndex int, expectedOutput []string) {
	TestContainerOutput(scenarioName, f.Client, pod, containerIndex, expectedOutput, f.Namespace.Name)
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

// CreateServiceForSimpleAppWithPods is a convenience wrapper to create a service and its matching pods all at once.
func (f *Framework) CreateServiceForSimpleAppWithPods(contPort int, svcPort int, appName string, podSpec func(n api.Node) api.PodSpec, count int, block bool) (error, *api.Service) {
	var err error = nil
	theService := f.CreateServiceForSimpleApp(contPort, svcPort, appName)
	f.CreatePodsPerNodeForSimpleApp(appName, podSpec, count)
	if block {
		err = WaitForPodsWithLabelRunning(f.Client, f.Namespace.Name, labels.SelectorFromSet(labels.Set(theService.Spec.Selector)))
	}
	return err, theService
}

// CreateServiceForSimpleApp returns a service that selects/exposes pods (send -1 ports if no exposure needed) with an app label.
func (f *Framework) CreateServiceForSimpleApp(contPort, svcPort int, appName string) *api.Service {
	if appName == "" {
		panic(fmt.Sprintf("no app name provided"))
	}

	serviceSelector := map[string]string{
		"app": appName + "-pod",
	}

	// For convenience, user sending ports are optional.
	portsFunc := func() []api.ServicePort {
		if contPort < 1 || svcPort < 1 {
			return nil
		} else {
			return []api.ServicePort{{
				Protocol:   "TCP",
				Port:       int32(svcPort),
				TargetPort: intstr.FromInt(contPort),
			}}
		}
	}
	Logf("Creating a service-for-%v for selecting app=%v-pod", appName, appName)
	service, err := f.Client.Services(f.Namespace.Name).Create(&api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: "service-for-" + appName,
			Labels: map[string]string{
				"app": appName + "-service",
			},
		},
		Spec: api.ServiceSpec{
			Ports:    portsFunc(),
			Selector: serviceSelector,
		},
	})
	ExpectNoError(err)
	return service
}

// CreatePodsPerNodeForSimpleApp Creates pods w/ labels.  Useful for tests which make a bunch of pods w/o any networking.
func (f *Framework) CreatePodsPerNodeForSimpleApp(appName string, podSpec func(n api.Node) api.PodSpec, maxCount int) map[string]string {
	nodes := ListSchedulableNodesOrDie(f.Client)
	labels := map[string]string{
		"app": appName + "-pod",
	}
	for i, node := range nodes.Items {
		// one per node, but no more than maxCount.
		if i <= maxCount {
			Logf("%v/%v : Creating container with label app=%v-pod", i, maxCount, appName)
			_, err := f.Client.Pods(f.Namespace.Name).Create(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name:   fmt.Sprintf(appName+"-pod-%v", i),
					Labels: labels,
				},
				Spec: podSpec(node),
			})
			ExpectNoError(err)
		}
	}
	return labels
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

	cmd := KubectlCmd(cmdArgs...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	Logf("Running '%s %s'", cmd.Path, strings.Join(cmd.Args, " "))
	err := cmd.Run()
	return stdout.Bytes(), stderr.Bytes(), err
}

// Wrapper function for ginkgo describe.  Adds namespacing.
// TODO: Support type safe tagging as well https://github.com/kubernetes/kubernetes/pull/22401.
func KubeDescribe(text string, body func()) bool {
	return Describe("[k8s.io] "+text, body)
}

// PodStateVerification represents a verification of pod state.
// Any time you have a set of pods that you want to operate against or query,
// this struct can be used to declaratively identify those pods.
type PodStateVerification struct {
	// Optional: only pods that have k=v labels will pass this filter.
	Selectors map[string]string

	// Required: The phases which are valid for your pod.
	ValidPhases []api.PodPhase

	// Optional: only pods passing this function will pass the filter
	// Verify a pod.
	// As an optimization, in addition to specfying filter (boolean),
	// this function allows specifying an error as well.
	// The error indicates that the polling of the pod spectrum should stop.
	Verify func(api.Pod) (bool, error)

	// Optional: only pods with this name will pass the filter.
	PodName string
}

type ClusterVerification struct {
	client    *client.Client
	namespace *api.Namespace // pointer rather than string, since ns isn't created until before each.
	podState  PodStateVerification
}

func (f *Framework) NewClusterVerification(filter PodStateVerification) *ClusterVerification {
	return &ClusterVerification{
		f.Client,
		f.Namespace,
		filter,
	}
}

func passesPodNameFilter(pod api.Pod, name string) bool {
	return name == "" || strings.Contains(pod.Name, name)
}

func passesVerifyFilter(pod api.Pod, verify func(p api.Pod) (bool, error)) (bool, error) {
	if verify == nil {
		return true, nil
	} else {
		verified, err := verify(pod)
		// If an error is returned, by definition, pod verification fails
		if err != nil {
			return false, err
		} else {
			return verified, nil
		}
	}
}

func passesPhasesFilter(pod api.Pod, validPhases []api.PodPhase) bool {
	passesPhaseFilter := false
	for _, phase := range validPhases {
		if pod.Status.Phase == phase {
			passesPhaseFilter = true
		}
	}
	return passesPhaseFilter
}

// filterLabels returns a list of pods which have labels.
func filterLabels(selectors map[string]string, cli *client.Client, ns string) (*api.PodList, error) {
	var err error
	var selector labels.Selector
	var pl *api.PodList
	// List pods based on selectors.  This might be a tiny optimization rather then filtering
	// everything manually.
	if len(selectors) > 0 {
		selector = labels.SelectorFromSet(labels.Set(selectors))
		options := api.ListOptions{LabelSelector: selector}
		pl, err = cli.Pods(ns).List(options)
	} else {
		pl, err = cli.Pods(ns).List(api.ListOptions{})
	}
	return pl, err
}

// filter filters pods which pass a filter.  It can be used to compose
// the more useful abstractions like ForEach, WaitFor, and so on, which
// can be used directly by tests.
func (p *PodStateVerification) filter(c *client.Client, namespace *api.Namespace) ([]api.Pod, error) {
	if len(p.ValidPhases) == 0 || namespace == nil {
		panic(fmt.Errorf("Need to specify a valid pod phases (%v) and namespace (%v). ", p.ValidPhases, namespace))
	}

	ns := namespace.Name
	pl, err := filterLabels(p.Selectors, c, ns) // Build an api.PodList to operate against.
	Logf("Selector matched %v pods for %v", len(pl.Items), p.Selectors)
	if len(pl.Items) == 0 || err != nil {
		return pl.Items, err
	}

	unfilteredPods := pl.Items
	filteredPods := []api.Pod{}
ReturnPodsSoFar:
	// Next: Pod must match at least one of the states that the user specified
	for _, pod := range unfilteredPods {
		if !(passesPhasesFilter(pod, p.ValidPhases) && passesPodNameFilter(pod, p.PodName)) {
			continue
		}
		passesVerify, err := passesVerifyFilter(pod, p.Verify)
		if err != nil {
			Logf("Error detected on %v : %v !", pod.Name, err)
			break ReturnPodsSoFar
		}
		if passesVerify {
			filteredPods = append(filteredPods, pod)
		}
	}
	return filteredPods, err
}

// WaitFor waits for some minimum number of pods to be verified, according to the PodStateVerification
// definition.
func (cl *ClusterVerification) WaitFor(atLeast int, timeout time.Duration) ([]api.Pod, error) {
	pods := []api.Pod{}
	var returnedErr error

	err := wait.Poll(1*time.Second, timeout, func() (bool, error) {
		pods, returnedErr = cl.podState.filter(cl.client, cl.namespace)

		// Failure
		if returnedErr != nil {
			Logf("Cutting polling short: We got an error from the pod filtering layer.")
			// stop polling if the pod filtering returns an error.  that should never happen.
			// it indicates, for example, that the client is broken or something non-pod related.
			return false, returnedErr
		}
		Logf("Found %v / %v", len(pods), atLeast)

		// Success
		if len(pods) >= atLeast {
			return true, nil
		}
		// Keep trying...
		return false, nil
	})
	Logf("WaitFor completed with timeout %v.  Pods found = %v out of %v", timeout, len(pods), atLeast)
	return pods, err
}

// WaitForOrFail provides a shorthand WaitFor with failure as an option if anything goes wrong.
func (cl *ClusterVerification) WaitForOrFail(atLeast int, timeout time.Duration) {
	pods, err := cl.WaitFor(atLeast, timeout)
	if err != nil || len(pods) < atLeast {
		Failf("Verified %v of %v pods , error : %v", len(pods), atLeast, err)
	}
}

// ForEach runs a function against every verifiable pod.  Be warned that this doesn't wait for "n" pods to verifiy,
// so it may return very quickly if you have strict pod state requirements.
//
// For example, if you require at least 5 pods to be running before your test will pass,
// its smart to first call "clusterVerification.WaitFor(5)" before you call clusterVerification.ForEach.
func (cl *ClusterVerification) ForEach(podFunc func(api.Pod)) error {
	pods, err := cl.podState.filter(cl.client, cl.namespace)
	if err == nil {
		Logf("ForEach: Found %v pods from the filter.  Now looping through them.", len(pods))
		for _, p := range pods {
			podFunc(p)
		}
	} else {
		Logf("ForEach: Something went wrong when filtering pods to execute against: %v", err)
	}

	return err
}
