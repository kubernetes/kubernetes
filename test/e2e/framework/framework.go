/*
Copyright 2015 The Kubernetes Authors.

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

// Package framework contains provider-independent helper code for
// building and running E2E tests with Ginkgo. The actual Ginkgo test
// suites gets assembled by combining this framework, the optional
// provider support code and specific tests via a separate .go file
// like Kubernetes' test/e2e.go.
package framework

import (
	"context"
	"fmt"
	"io/ioutil"
	"math/rand"
	"path"
	"strings"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/runtime"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/discovery"
	cacheddiscovery "k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	scaleclient "k8s.io/client-go/scale"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	// TODO: Remove the following imports (ref: https://github.com/kubernetes/kubernetes/issues/81245)
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
)

const (
	// DefaultNamespaceDeletionTimeout is timeout duration for waiting for a namespace deletion.
	DefaultNamespaceDeletionTimeout = 5 * time.Minute
)

// Framework supports common operations used by e2e tests; it will keep a client & a namespace for you.
// Eventual goal is to merge this with integration test framework.
type Framework struct {
	BaseName string

	// Set together with creating the ClientSet and the namespace.
	// Guaranteed to be unique in the cluster even when running the same
	// test multiple times in parallel.
	UniqueName string

	clientConfig                     *rest.Config
	ClientSet                        clientset.Interface
	KubemarkExternalClusterClientSet clientset.Interface

	DynamicClient dynamic.Interface

	ScalesGetter scaleclient.ScalesGetter

	SkipNamespaceCreation    bool            // Whether to skip creating a namespace
	Namespace                *v1.Namespace   // Every test has at least one namespace unless creation is skipped
	namespacesToDelete       []*v1.Namespace // Some tests have more than one.
	NamespaceDeletionTimeout time.Duration
	SkipPrivilegedPSPBinding bool // Whether to skip creating a binding to the privileged PSP in the test namespace

	gatherer *ContainerResourceGatherer
	// Constraints that passed to a check which is executed after data is gathered to
	// see if 99% of results are within acceptable bounds. It has to be injected in the test,
	// as expectations vary greatly. Constraints are grouped by the container names.
	AddonResourceConstraints map[string]ResourceConstraint

	logsSizeWaitGroup    sync.WaitGroup
	logsSizeCloseChannel chan bool
	logsSizeVerifier     *LogsSizeVerifier

	// Flaky operation failures in an e2e test can be captured through this.
	flakeReport *FlakeReport

	// To make sure that this framework cleans up after itself, no matter what,
	// we install a Cleanup action before each test and clear it after.  If we
	// should abort, the AfterSuite hook should run all Cleanup actions.
	cleanupHandle CleanupActionHandle

	// afterEaches is a map of name to function to be called after each test.  These are not
	// cleared.  The call order is randomized so that no dependencies can grow between
	// the various afterEaches
	afterEaches map[string]AfterEachActionFunc

	// beforeEachStarted indicates that BeforeEach has started
	beforeEachStarted bool

	// configuration for framework's client
	Options Options

	// Place where various additional data is stored during test run to be printed to ReportDir,
	// or stdout if ReportDir is not set once test ends.
	TestSummaries []TestDataSummary

	// Place to keep ClusterAutoscaler metrics from before test in order to compute delta.
	clusterAutoscalerMetricsBeforeTest e2emetrics.Collection

	// Timeouts contains the custom timeouts used during the test execution.
	Timeouts *TimeoutContext
}

// AfterEachActionFunc is a function that can be called after each test
type AfterEachActionFunc func(f *Framework, failed bool)

// TestDataSummary is an interface for managing test data.
type TestDataSummary interface {
	SummaryKind() string
	PrintHumanReadable() string
	PrintJSON() string
}

// Options is a struct for managing test framework options.
type Options struct {
	ClientQPS    float32
	ClientBurst  int
	GroupVersion *schema.GroupVersion
}

// NewFrameworkWithCustomTimeouts makes a framework with with custom timeouts.
func NewFrameworkWithCustomTimeouts(baseName string, timeouts *TimeoutContext) *Framework {
	f := NewDefaultFramework(baseName)
	f.Timeouts = timeouts
	return f
}

// NewDefaultFramework makes a new framework and sets up a BeforeEach/AfterEach for
// you (you can write additional before/after each functions).
func NewDefaultFramework(baseName string) *Framework {
	options := Options{
		ClientQPS:   20,
		ClientBurst: 50,
	}
	return NewFramework(baseName, options, nil)
}

// NewFramework creates a test framework.
func NewFramework(baseName string, options Options, client clientset.Interface) *Framework {
	f := &Framework{
		BaseName:                 baseName,
		AddonResourceConstraints: make(map[string]ResourceConstraint),
		Options:                  options,
		ClientSet:                client,
		Timeouts:                 NewTimeoutContextWithDefaults(),
	}

	f.AddAfterEach("dumpNamespaceInfo", func(f *Framework, failed bool) {
		if !failed {
			return
		}
		if !TestContext.DumpLogsOnFailure {
			return
		}
		if !f.SkipNamespaceCreation {
			for _, ns := range f.namespacesToDelete {
				DumpAllNamespaceInfo(f.ClientSet, ns.Name)
			}
		}
	})

	ginkgo.BeforeEach(f.BeforeEach)
	ginkgo.AfterEach(f.AfterEach)

	return f
}

// BeforeEach gets a client and makes a namespace.
func (f *Framework) BeforeEach() {
	f.beforeEachStarted = true

	// The fact that we need this feels like a bug in ginkgo.
	// https://github.com/onsi/ginkgo/issues/222
	f.cleanupHandle = AddCleanupAction(f.AfterEach)
	if f.ClientSet == nil {
		ginkgo.By("Creating a kubernetes client")
		config, err := LoadConfig()
		ExpectNoError(err)

		config.QPS = f.Options.ClientQPS
		config.Burst = f.Options.ClientBurst
		if f.Options.GroupVersion != nil {
			config.GroupVersion = f.Options.GroupVersion
		}
		if TestContext.KubeAPIContentType != "" {
			config.ContentType = TestContext.KubeAPIContentType
		}
		f.clientConfig = rest.CopyConfig(config)
		f.ClientSet, err = clientset.NewForConfig(config)
		ExpectNoError(err)
		f.DynamicClient, err = dynamic.NewForConfig(config)
		ExpectNoError(err)

		// create scales getter, set GroupVersion and NegotiatedSerializer to default values
		// as they are required when creating a REST client.
		if config.GroupVersion == nil {
			config.GroupVersion = &schema.GroupVersion{}
		}
		if config.NegotiatedSerializer == nil {
			config.NegotiatedSerializer = scheme.Codecs
		}
		restClient, err := rest.RESTClientFor(config)
		ExpectNoError(err)
		discoClient, err := discovery.NewDiscoveryClientForConfig(config)
		ExpectNoError(err)
		cachedDiscoClient := cacheddiscovery.NewMemCacheClient(discoClient)
		restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cachedDiscoClient)
		restMapper.Reset()
		resolver := scaleclient.NewDiscoveryScaleKindResolver(cachedDiscoClient)
		f.ScalesGetter = scaleclient.New(restClient, restMapper, dynamic.LegacyAPIPathResolverFunc, resolver)

		TestContext.CloudConfig.Provider.FrameworkBeforeEach(f)
	}

	if !f.SkipNamespaceCreation {
		ginkgo.By(fmt.Sprintf("Building a namespace api object, basename %s", f.BaseName))
		namespace, err := f.CreateNamespace(f.BaseName, map[string]string{
			"e2e-framework": f.BaseName,
		})
		ExpectNoError(err)

		f.Namespace = namespace

		if TestContext.VerifyServiceAccount {
			ginkgo.By("Waiting for a default service account to be provisioned in namespace")
			err = WaitForDefaultServiceAccountInNamespace(f.ClientSet, namespace.Name)
			ExpectNoError(err)
		} else {
			Logf("Skipping waiting for service account")
		}
		f.UniqueName = f.Namespace.GetName()
	} else {
		// not guaranteed to be unique, but very likely
		f.UniqueName = fmt.Sprintf("%s-%08x", f.BaseName, rand.Int31())
	}

	if TestContext.GatherKubeSystemResourceUsageData != "false" && TestContext.GatherKubeSystemResourceUsageData != "none" {
		var err error
		var nodeMode NodesSet
		switch TestContext.GatherKubeSystemResourceUsageData {
		case "master":
			nodeMode = MasterNodes
		case "masteranddns":
			nodeMode = MasterAndDNSNodes
		default:
			nodeMode = AllNodes
		}

		f.gatherer, err = NewResourceUsageGatherer(f.ClientSet, ResourceGathererOptions{
			InKubemark:                  ProviderIs("kubemark"),
			Nodes:                       nodeMode,
			ResourceDataGatheringPeriod: 60 * time.Second,
			ProbeDuration:               15 * time.Second,
			PrintVerboseLogs:            false,
		}, nil)
		if err != nil {
			Logf("Error while creating NewResourceUsageGatherer: %v", err)
		} else {
			go f.gatherer.StartGatheringData()
		}
	}

	if TestContext.GatherLogsSizes {
		f.logsSizeWaitGroup = sync.WaitGroup{}
		f.logsSizeWaitGroup.Add(1)
		f.logsSizeCloseChannel = make(chan bool)
		f.logsSizeVerifier = NewLogsVerifier(f.ClientSet, f.logsSizeCloseChannel)
		go func() {
			f.logsSizeVerifier.Run()
			f.logsSizeWaitGroup.Done()
		}()
	}

	gatherMetricsAfterTest := TestContext.GatherMetricsAfterTest == "true" || TestContext.GatherMetricsAfterTest == "master"
	if gatherMetricsAfterTest && TestContext.IncludeClusterAutoscalerMetrics {
		grabber, err := e2emetrics.NewMetricsGrabber(f.ClientSet, f.KubemarkExternalClusterClientSet, f.ClientConfig(), !ProviderIs("kubemark"), false, false, false, TestContext.IncludeClusterAutoscalerMetrics, false)
		if err != nil {
			Logf("Failed to create MetricsGrabber (skipping ClusterAutoscaler metrics gathering before test): %v", err)
		} else {
			f.clusterAutoscalerMetricsBeforeTest, err = grabber.Grab()
			if err != nil {
				Logf("MetricsGrabber failed to grab CA metrics before test (skipping metrics gathering): %v", err)
			} else {
				Logf("Gathered ClusterAutoscaler metrics before test")
			}
		}

	}

	f.flakeReport = NewFlakeReport()
}

// printSummaries prints summaries of tests.
func printSummaries(summaries []TestDataSummary, testBaseName string) {
	now := time.Now()
	for i := range summaries {
		Logf("Printing summary: %v", summaries[i].SummaryKind())
		switch TestContext.OutputPrintType {
		case "hr":
			if TestContext.ReportDir == "" {
				Logf(summaries[i].PrintHumanReadable())
			} else {
				// TODO: learn to extract test name and append it to the kind instead of timestamp.
				filePath := path.Join(TestContext.ReportDir, summaries[i].SummaryKind()+"_"+testBaseName+"_"+now.Format(time.RFC3339)+".txt")
				if err := ioutil.WriteFile(filePath, []byte(summaries[i].PrintHumanReadable()), 0644); err != nil {
					Logf("Failed to write file %v with test performance data: %v", filePath, err)
				}
			}
		case "json":
			fallthrough
		default:
			if TestContext.OutputPrintType != "json" {
				Logf("Unknown output type: %v. Printing JSON", TestContext.OutputPrintType)
			}
			if TestContext.ReportDir == "" {
				Logf("%v JSON\n%v", summaries[i].SummaryKind(), summaries[i].PrintJSON())
				Logf("Finished")
			} else {
				// TODO: learn to extract test name and append it to the kind instead of timestamp.
				filePath := path.Join(TestContext.ReportDir, summaries[i].SummaryKind()+"_"+testBaseName+"_"+now.Format(time.RFC3339)+".json")
				Logf("Writing to %s", filePath)
				if err := ioutil.WriteFile(filePath, []byte(summaries[i].PrintJSON()), 0644); err != nil {
					Logf("Failed to write file %v with test performance data: %v", filePath, err)
				}
			}
		}
	}
}

// AddAfterEach is a way to add a function to be called after every test.  The execution order is intentionally random
// to avoid growing dependencies.  If you register the same name twice, it is a coding error and will panic.
func (f *Framework) AddAfterEach(name string, fn AfterEachActionFunc) {
	if _, ok := f.afterEaches[name]; ok {
		panic(fmt.Sprintf("%q is already registered", name))
	}

	if f.afterEaches == nil {
		f.afterEaches = map[string]AfterEachActionFunc{}
	}
	f.afterEaches[name] = fn
}

// AfterEach deletes the namespace, after reading its events.
func (f *Framework) AfterEach() {
	// If BeforeEach never started AfterEach should be skipped.
	// Currently some tests under e2e/storage have this condition.
	if !f.beforeEachStarted {
		return
	}

	RemoveCleanupAction(f.cleanupHandle)

	// This should not happen. Given ClientSet is a public field a test must have updated it!
	// Error out early before any API calls during cleanup.
	if f.ClientSet == nil {
		Failf("The framework ClientSet must not be nil at this point")
	}

	// DeleteNamespace at the very end in defer, to avoid any
	// expectation failures preventing deleting the namespace.
	defer func() {
		nsDeletionErrors := map[string]error{}
		// Whether to delete namespace is determined by 3 factors: delete-namespace flag, delete-namespace-on-failure flag and the test result
		// if delete-namespace set to false, namespace will always be preserved.
		// if delete-namespace is true and delete-namespace-on-failure is false, namespace will be preserved if test failed.
		if TestContext.DeleteNamespace && (TestContext.DeleteNamespaceOnFailure || !ginkgo.CurrentGinkgoTestDescription().Failed) {
			for _, ns := range f.namespacesToDelete {
				ginkgo.By(fmt.Sprintf("Destroying namespace %q for this suite.", ns.Name))
				if err := f.ClientSet.CoreV1().Namespaces().Delete(context.TODO(), ns.Name, metav1.DeleteOptions{}); err != nil {
					if !apierrors.IsNotFound(err) {
						nsDeletionErrors[ns.Name] = err

						// Dump namespace if we are unable to delete the namespace and the dump was not already performed.
						if !ginkgo.CurrentGinkgoTestDescription().Failed && TestContext.DumpLogsOnFailure {
							DumpAllNamespaceInfo(f.ClientSet, ns.Name)
						}
					} else {
						Logf("Namespace %v was already deleted", ns.Name)
					}
				}
			}
		} else {
			if !TestContext.DeleteNamespace {
				Logf("Found DeleteNamespace=false, skipping namespace deletion!")
			} else {
				Logf("Found DeleteNamespaceOnFailure=false and current test failed, skipping namespace deletion!")
			}
		}

		// Paranoia-- prevent reuse!
		f.Namespace = nil
		f.clientConfig = nil
		f.ClientSet = nil
		f.namespacesToDelete = nil

		// if we had errors deleting, report them now.
		if len(nsDeletionErrors) != 0 {
			messages := []string{}
			for namespaceKey, namespaceErr := range nsDeletionErrors {
				messages = append(messages, fmt.Sprintf("Couldn't delete ns: %q: %s (%#v)", namespaceKey, namespaceErr, namespaceErr))
			}
			Failf(strings.Join(messages, ","))
		}
	}()

	// run all aftereach functions in random order to ensure no dependencies grow
	for _, afterEachFn := range f.afterEaches {
		afterEachFn(f, ginkgo.CurrentGinkgoTestDescription().Failed)
	}

	if TestContext.GatherKubeSystemResourceUsageData != "false" && TestContext.GatherKubeSystemResourceUsageData != "none" && f.gatherer != nil {
		ginkgo.By("Collecting resource usage data")
		summary, resourceViolationError := f.gatherer.StopAndSummarize([]int{90, 99, 100}, f.AddonResourceConstraints)
		defer ExpectNoError(resourceViolationError)
		f.TestSummaries = append(f.TestSummaries, summary)
	}

	if TestContext.GatherLogsSizes {
		ginkgo.By("Gathering log sizes data")
		close(f.logsSizeCloseChannel)
		f.logsSizeWaitGroup.Wait()
		f.TestSummaries = append(f.TestSummaries, f.logsSizeVerifier.GetSummary())
	}

	if TestContext.GatherMetricsAfterTest != "false" {
		ginkgo.By("Gathering metrics")
		// Grab apiserver, scheduler, controller-manager metrics and (optionally) nodes' kubelet metrics.
		grabMetricsFromKubelets := TestContext.GatherMetricsAfterTest != "master" && !ProviderIs("kubemark")
		grabber, err := e2emetrics.NewMetricsGrabber(f.ClientSet, f.KubemarkExternalClusterClientSet, f.ClientConfig(), grabMetricsFromKubelets, true, true, true, TestContext.IncludeClusterAutoscalerMetrics, false)
		if err != nil {
			Logf("Failed to create MetricsGrabber (skipping metrics gathering): %v", err)
		} else {
			received, err := grabber.Grab()
			if err != nil {
				Logf("MetricsGrabber failed to grab some of the metrics: %v", err)
			}
			(*e2emetrics.ComponentCollection)(&received).ComputeClusterAutoscalerMetricsDelta(f.clusterAutoscalerMetricsBeforeTest)
			f.TestSummaries = append(f.TestSummaries, (*e2emetrics.ComponentCollection)(&received))
		}
	}

	TestContext.CloudConfig.Provider.FrameworkAfterEach(f)

	// Report any flakes that were observed in the e2e test and reset.
	if f.flakeReport != nil && f.flakeReport.GetFlakeCount() > 0 {
		f.TestSummaries = append(f.TestSummaries, f.flakeReport)
		f.flakeReport = nil
	}

	printSummaries(f.TestSummaries, f.BaseName)

	// Check whether all nodes are ready after the test.
	// This is explicitly done at the very end of the test, to avoid
	// e.g. not removing namespace in case of this failure.
	if err := AllNodesReady(f.ClientSet, 3*time.Minute); err != nil {
		Failf("All nodes should be ready after test, %v", err)
	}
}

// DeleteNamespace can be used to delete a namespace. Additionally it can be used to
// dump namespace information so as it can be used as an alternative of framework
// deleting the namespace towards the end.
func (f *Framework) DeleteNamespace(name string) {
	defer func() {
		err := f.ClientSet.CoreV1().Namespaces().Delete(context.TODO(), name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			Logf("error deleting namespace %s: %v", name, err)
			return
		}
		err = WaitForNamespacesDeleted(f.ClientSet, []string{name}, DefaultNamespaceDeletionTimeout)
		if err != nil {
			Logf("error deleting namespace %s: %v", name, err)
			return
		}
		// remove deleted namespace from namespacesToDelete map
		for i, ns := range f.namespacesToDelete {
			if ns == nil {
				continue
			}
			if ns.Name == name {
				f.namespacesToDelete = append(f.namespacesToDelete[:i], f.namespacesToDelete[i+1:]...)
			}
		}
	}()
	// if current test failed then we should dump namespace information
	if !f.SkipNamespaceCreation && ginkgo.CurrentGinkgoTestDescription().Failed && TestContext.DumpLogsOnFailure {
		DumpAllNamespaceInfo(f.ClientSet, name)
	}

}

// CreateNamespace creates a namespace for e2e testing.
func (f *Framework) CreateNamespace(baseName string, labels map[string]string) (*v1.Namespace, error) {
	createTestingNS := TestContext.CreateTestingNS
	if createTestingNS == nil {
		createTestingNS = CreateTestingNS
	}
	ns, err := createTestingNS(baseName, f.ClientSet, labels)
	// check ns instead of err to see if it's nil as we may
	// fail to create serviceAccount in it.
	f.AddNamespacesToDelete(ns)

	if err == nil && !f.SkipPrivilegedPSPBinding {
		CreatePrivilegedPSPBinding(f.ClientSet, ns.Name)
	}

	return ns, err
}

// RecordFlakeIfError records flakeness info if error happens.
// NOTE: This function is not used at any places yet, but we are in progress for https://github.com/kubernetes/kubernetes/issues/66239 which requires this. Please don't remove this.
func (f *Framework) RecordFlakeIfError(err error, optionalDescription ...interface{}) {
	f.flakeReport.RecordFlakeIfError(err, optionalDescription)
}

// AddNamespacesToDelete adds one or more namespaces to be deleted when the test
// completes.
func (f *Framework) AddNamespacesToDelete(namespaces ...*v1.Namespace) {
	for _, ns := range namespaces {
		if ns == nil {
			continue
		}
		f.namespacesToDelete = append(f.namespacesToDelete, ns)

	}
}

// ClientConfig an externally accessible method for reading the kube client config.
func (f *Framework) ClientConfig() *rest.Config {
	ret := rest.CopyConfig(f.clientConfig)
	// json is least common denominator
	ret.ContentType = runtime.ContentTypeJSON
	ret.AcceptContentTypes = runtime.ContentTypeJSON
	return ret
}

// TestContainerOutput runs the given pod in the given namespace and waits
// for all of the containers in the podSpec to move into the 'Success' status, and tests
// the specified container log against the given expected output using a substring matcher.
func (f *Framework) TestContainerOutput(scenarioName string, pod *v1.Pod, containerIndex int, expectedOutput []string) {
	f.testContainerOutputMatcher(scenarioName, pod, containerIndex, expectedOutput, gomega.ContainSubstring)
}

// TestContainerOutputRegexp runs the given pod in the given namespace and waits
// for all of the containers in the podSpec to move into the 'Success' status, and tests
// the specified container log against the given expected output using a regexp matcher.
func (f *Framework) TestContainerOutputRegexp(scenarioName string, pod *v1.Pod, containerIndex int, expectedOutput []string) {
	f.testContainerOutputMatcher(scenarioName, pod, containerIndex, expectedOutput, gomega.MatchRegexp)
}

// KubeUser is a struct for managing kubernetes user info.
type KubeUser struct {
	Name string `yaml:"name"`
	User struct {
		Username string `yaml:"username"`
		Password string `yaml:"password" datapolicy:"password"`
		Token    string `yaml:"token" datapolicy:"token"`
	} `yaml:"user"`
}

// KubeCluster is a struct for managing kubernetes cluster info.
type KubeCluster struct {
	Name    string `yaml:"name"`
	Cluster struct {
		CertificateAuthorityData string `yaml:"certificate-authority-data"`
		Server                   string `yaml:"server"`
	} `yaml:"cluster"`
}

// KubeConfig is a struct for managing kubernetes config.
type KubeConfig struct {
	Contexts []struct {
		Name    string `yaml:"name"`
		Context struct {
			Cluster string `yaml:"cluster"`
			User    string
		} `yaml:"context"`
	} `yaml:"contexts"`

	Clusters []KubeCluster `yaml:"clusters"`

	Users []KubeUser `yaml:"users"`
}

// FindUser returns user info which is the specified user name.
func (kc *KubeConfig) FindUser(name string) *KubeUser {
	for _, user := range kc.Users {
		if user.Name == name {
			return &user
		}
	}
	return nil
}

// FindCluster returns cluster info which is the specified cluster name.
func (kc *KubeConfig) FindCluster(name string) *KubeCluster {
	for _, cluster := range kc.Clusters {
		if cluster.Name == name {
			return &cluster
		}
	}
	return nil
}

// ConformanceIt is wrapper function for ginkgo It.  Adds "[Conformance]" tag and makes static analysis easier.
func ConformanceIt(text string, body interface{}, timeout ...float64) bool {
	return ginkgo.It(text+" [Conformance]", body, timeout...)
}

// PodStateVerification represents a verification of pod state.
// Any time you have a set of pods that you want to operate against or query,
// this struct can be used to declaratively identify those pods.
type PodStateVerification struct {
	// Optional: only pods that have k=v labels will pass this filter.
	Selectors map[string]string

	// Required: The phases which are valid for your pod.
	ValidPhases []v1.PodPhase

	// Optional: only pods passing this function will pass the filter
	// Verify a pod.
	// As an optimization, in addition to specifying filter (boolean),
	// this function allows specifying an error as well.
	// The error indicates that the polling of the pod spectrum should stop.
	Verify func(v1.Pod) (bool, error)

	// Optional: only pods with this name will pass the filter.
	PodName string
}

// ClusterVerification is a struct for a verification of cluster state.
type ClusterVerification struct {
	client    clientset.Interface
	namespace *v1.Namespace // pointer rather than string, since ns isn't created until before each.
	podState  PodStateVerification
}

// NewClusterVerification creates a new cluster verification.
func (f *Framework) NewClusterVerification(namespace *v1.Namespace, filter PodStateVerification) *ClusterVerification {
	return &ClusterVerification{
		f.ClientSet,
		namespace,
		filter,
	}
}

func passesPodNameFilter(pod v1.Pod, name string) bool {
	return name == "" || strings.Contains(pod.Name, name)
}

func passesVerifyFilter(pod v1.Pod, verify func(p v1.Pod) (bool, error)) (bool, error) {
	if verify == nil {
		return true, nil
	}

	verified, err := verify(pod)
	// If an error is returned, by definition, pod verification fails
	if err != nil {
		return false, err
	}
	return verified, nil
}

func passesPhasesFilter(pod v1.Pod, validPhases []v1.PodPhase) bool {
	passesPhaseFilter := false
	for _, phase := range validPhases {
		if pod.Status.Phase == phase {
			passesPhaseFilter = true
		}
	}
	return passesPhaseFilter
}

// filterLabels returns a list of pods which have labels.
func filterLabels(selectors map[string]string, cli clientset.Interface, ns string) (*v1.PodList, error) {
	var err error
	var selector labels.Selector
	var pl *v1.PodList
	// List pods based on selectors.  This might be a tiny optimization rather then filtering
	// everything manually.
	if len(selectors) > 0 {
		selector = labels.SelectorFromSet(labels.Set(selectors))
		options := metav1.ListOptions{LabelSelector: selector.String()}
		pl, err = cli.CoreV1().Pods(ns).List(context.TODO(), options)
	} else {
		pl, err = cli.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{})
	}
	return pl, err
}

// filter filters pods which pass a filter.  It can be used to compose
// the more useful abstractions like ForEach, WaitFor, and so on, which
// can be used directly by tests.
func (p *PodStateVerification) filter(c clientset.Interface, namespace *v1.Namespace) ([]v1.Pod, error) {
	if len(p.ValidPhases) == 0 || namespace == nil {
		panic(fmt.Errorf("Need to specify a valid pod phases (%v) and namespace (%v). ", p.ValidPhases, namespace))
	}

	ns := namespace.Name
	pl, err := filterLabels(p.Selectors, c, ns) // Build an v1.PodList to operate against.
	Logf("Selector matched %v pods for %v", len(pl.Items), p.Selectors)
	if len(pl.Items) == 0 || err != nil {
		return pl.Items, err
	}

	unfilteredPods := pl.Items
	filteredPods := []v1.Pod{}
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
func (cl *ClusterVerification) WaitFor(atLeast int, timeout time.Duration) ([]v1.Pod, error) {
	pods := []v1.Pod{}
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

// ForEach runs a function against every verifiable pod.  Be warned that this doesn't wait for "n" pods to verify,
// so it may return very quickly if you have strict pod state requirements.
//
// For example, if you require at least 5 pods to be running before your test will pass,
// its smart to first call "clusterVerification.WaitFor(5)" before you call clusterVerification.ForEach.
func (cl *ClusterVerification) ForEach(podFunc func(v1.Pod)) error {
	pods, err := cl.podState.filter(cl.client, cl.namespace)
	if err == nil {
		if len(pods) == 0 {
			Failf("No pods matched the filter.")
		}
		Logf("ForEach: Found %v pods from the filter.  Now looping through them.", len(pods))
		for _, p := range pods {
			podFunc(p)
		}
	} else {
		Logf("ForEach: Something went wrong when filtering pods to execute against: %v", err)
	}

	return err
}
