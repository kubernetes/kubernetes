/*
Copyright 2014 The Kubernetes Authors.

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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/net/websocket"

	"k8s.io/klog"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	gomegatypes "github.com/onsi/gomega/types"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	restclient "k8s.io/client-go/rest"
	scaleclient "k8s.io/client-go/scale"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	watchtools "k8s.io/client-go/tools/watch"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/client/conditions"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/master/ports"
	taintutils "k8s.io/kubernetes/pkg/util/taints"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	uexec "k8s.io/utils/exec"

	// TODO: Remove the following imports (ref: https://github.com/kubernetes/kubernetes/issues/81245)
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eresource "k8s.io/kubernetes/test/e2e/framework/resource"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
)

const (
	// PodListTimeout is how long to wait for the pod to be listable.
	PodListTimeout = time.Minute
	// PodStartTimeout is how long to wait for the pod to be started.
	PodStartTimeout = 5 * time.Minute

	// PodStartShortTimeout is same as `PodStartTimeout` to wait for the pod to be started, but shorter.
	// Use it case by case when we are sure pod start will not be delayed.
	// minutes by slow docker pulls or something else.
	PodStartShortTimeout = 2 * time.Minute

	// PodDeleteTimeout is how long to wait for a pod to be deleted.
	PodDeleteTimeout = 5 * time.Minute

	// PodGetTimeout is how long to wait for a pod to be got.
	PodGetTimeout = 2 * time.Minute

	// PodEventTimeout is how much we wait for a pod event to occur.
	PodEventTimeout = 2 * time.Minute

	// ServiceStartTimeout is how long to wait for a service endpoint to be resolvable.
	ServiceStartTimeout = 3 * time.Minute

	// Poll is how often to Poll pods, nodes and claims.
	Poll = 2 * time.Second

	// PollShortTimeout is the short timeout value in polling.
	PollShortTimeout = 1 * time.Minute

	// ServiceAccountProvisionTimeout is how long to wait for a service account to be provisioned.
	// service accounts are provisioned after namespace creation
	// a service account is required to support pod creation in a namespace as part of admission control
	ServiceAccountProvisionTimeout = 2 * time.Minute

	// SingleCallTimeout is how long to try single API calls (like 'get' or 'list'). Used to prevent
	// transient failures from failing tests.
	SingleCallTimeout = 5 * time.Minute

	// NodeReadyInitialTimeout is how long nodes have to be "ready" when a test begins. They should already
	// be "ready" before the test starts, so this is small.
	NodeReadyInitialTimeout = 20 * time.Second

	// PodReadyBeforeTimeout is how long pods have to be "ready" when a test begins.
	PodReadyBeforeTimeout = 5 * time.Minute

	// ClaimProvisionShortTimeout is same as `ClaimProvisionTimeout` to wait for claim to be dynamically provisioned, but shorter.
	// Use it case by case when we are sure this timeout is enough.
	ClaimProvisionShortTimeout = 1 * time.Minute

	// ClaimProvisionTimeout is how long claims have to become dynamically provisioned.
	ClaimProvisionTimeout = 5 * time.Minute

	// RestartNodeReadyAgainTimeout is how long a node is allowed to become "Ready" after it is restarted before
	// the test is considered failed.
	RestartNodeReadyAgainTimeout = 5 * time.Minute

	// RestartPodReadyAgainTimeout is how long a pod is allowed to become "running" and "ready" after a node
	// restart before test is considered failed.
	RestartPodReadyAgainTimeout = 5 * time.Minute

	// SnapshotCreateTimeout is how long for snapshot to create snapshotContent.
	SnapshotCreateTimeout = 5 * time.Minute

	// Number of objects that gc can delete in a second.
	// GC issues 2 requestes for single delete.
	gcThroughput = 10

	// Minimal number of nodes for the cluster to be considered large.
	largeClusterThreshold = 100

	// TODO(justinsb): Avoid hardcoding this.
	awsMasterIP = "172.20.0.9"

	// ssh port
	sshPort = "22"
)

var (
	// BusyBoxImage is the image URI of BusyBox.
	BusyBoxImage = imageutils.GetE2EImage(imageutils.BusyBox)

	// AgnHostImage is the image URI of AgnHost
	AgnHostImage = imageutils.GetE2EImage(imageutils.Agnhost)

	// ProvidersWithSSH are those providers where each node is accessible with SSH
	ProvidersWithSSH = []string{"gce", "gke", "aws", "local"}

	// ServeHostnameImage is a serve hostname image name.
	ServeHostnameImage = imageutils.GetE2EImage(imageutils.Agnhost)
)

// RunID is a unique identifier of the e2e run.
// Beware that this ID is not the same for all tests in the e2e run, because each Ginkgo node creates it separately.
var RunID = uuid.NewUUID()

// CreateTestingNSFn is a func that is responsible for creating namespace used for executing e2e tests.
type CreateTestingNSFn func(baseName string, c clientset.Interface, labels map[string]string) (*v1.Namespace, error)

// GetMasterHost returns a hostname of a master.
func GetMasterHost() string {
	masterURL, err := url.Parse(TestContext.Host)
	ExpectNoError(err)
	return masterURL.Hostname()
}

// ProviderIs returns true if the provider is included is the providers. Otherwise false.
func ProviderIs(providers ...string) bool {
	for _, provider := range providers {
		if strings.EqualFold(provider, TestContext.Provider) {
			return true
		}
	}
	return false
}

// MasterOSDistroIs returns true if the master OS distro is included in the supportedMasterOsDistros. Otherwise false.
func MasterOSDistroIs(supportedMasterOsDistros ...string) bool {
	for _, distro := range supportedMasterOsDistros {
		if strings.EqualFold(distro, TestContext.MasterOSDistro) {
			return true
		}
	}
	return false
}

// NodeOSDistroIs returns true if the node OS distro is included in the supportedNodeOsDistros. Otherwise false.
func NodeOSDistroIs(supportedNodeOsDistros ...string) bool {
	for _, distro := range supportedNodeOsDistros {
		if strings.EqualFold(distro, TestContext.NodeOSDistro) {
			return true
		}
	}
	return false
}

// DeleteNamespaces deletes all namespaces that match the given delete and skip filters.
// Filter is by simple strings.Contains; first skip filter, then delete filter.
// Returns the list of deleted namespaces or an error.
func DeleteNamespaces(c clientset.Interface, deleteFilter, skipFilter []string) ([]string, error) {
	ginkgo.By("Deleting namespaces")
	nsList, err := c.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{})
	ExpectNoError(err, "Failed to get namespace list")
	var deleted []string
	var wg sync.WaitGroup
OUTER:
	for _, item := range nsList.Items {
		if skipFilter != nil {
			for _, pattern := range skipFilter {
				if strings.Contains(item.Name, pattern) {
					continue OUTER
				}
			}
		}
		if deleteFilter != nil {
			var shouldDelete bool
			for _, pattern := range deleteFilter {
				if strings.Contains(item.Name, pattern) {
					shouldDelete = true
					break
				}
			}
			if !shouldDelete {
				continue OUTER
			}
		}
		wg.Add(1)
		deleted = append(deleted, item.Name)
		go func(nsName string) {
			defer wg.Done()
			defer ginkgo.GinkgoRecover()
			gomega.Expect(c.CoreV1().Namespaces().Delete(context.TODO(), nsName, metav1.DeleteOptions{})).To(gomega.Succeed())
			Logf("namespace : %v api call to delete is complete ", nsName)
		}(item.Name)
	}
	wg.Wait()
	return deleted, nil
}

// WaitForNamespacesDeleted waits for the namespaces to be deleted.
func WaitForNamespacesDeleted(c clientset.Interface, namespaces []string, timeout time.Duration) error {
	ginkgo.By("Waiting for namespaces to vanish")
	nsMap := map[string]bool{}
	for _, ns := range namespaces {
		nsMap[ns] = true
	}
	//Now POLL until all namespaces have been eradicated.
	return wait.Poll(2*time.Second, timeout,
		func() (bool, error) {
			nsList, err := c.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{})
			if err != nil {
				return false, err
			}
			for _, item := range nsList.Items {
				if _, ok := nsMap[item.Name]; ok {
					return false, nil
				}
			}
			return true, nil
		})
}

func waitForServiceAccountInNamespace(c clientset.Interface, ns, serviceAccountName string, timeout time.Duration) error {
	w, err := c.CoreV1().ServiceAccounts(ns).Watch(context.TODO(), metav1.SingleObject(metav1.ObjectMeta{Name: serviceAccountName}))
	if err != nil {
		return err
	}
	ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), timeout)
	defer cancel()
	_, err = watchtools.UntilWithoutRetry(ctx, w, conditions.ServiceAccountHasSecrets)
	return err
}

// WaitForDefaultServiceAccountInNamespace waits for the default service account to be provisioned
// the default service account is what is associated with pods when they do not specify a service account
// as a result, pods are not able to be provisioned in a namespace until the service account is provisioned
func WaitForDefaultServiceAccountInNamespace(c clientset.Interface, namespace string) error {
	return waitForServiceAccountInNamespace(c, namespace, "default", ServiceAccountProvisionTimeout)
}

// WaitForPersistentVolumeDeleted waits for a PersistentVolume to get deleted or until timeout occurs, whichever comes first.
func WaitForPersistentVolumeDeleted(c clientset.Interface, pvName string, Poll, timeout time.Duration) error {
	Logf("Waiting up to %v for PersistentVolume %s to get deleted", timeout, pvName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pv, err := c.CoreV1().PersistentVolumes().Get(context.TODO(), pvName, metav1.GetOptions{})
		if err == nil {
			Logf("PersistentVolume %s found and phase=%s (%v)", pvName, pv.Status.Phase, time.Since(start))
			continue
		}
		if apierrors.IsNotFound(err) {
			Logf("PersistentVolume %s was removed", pvName)
			return nil
		}
		Logf("Get persistent volume %s in failed, ignoring for %v: %v", pvName, Poll, err)
	}
	return fmt.Errorf("PersistentVolume %s still exists within %v", pvName, timeout)
}

// findAvailableNamespaceName random namespace name starting with baseName.
func findAvailableNamespaceName(baseName string, c clientset.Interface) (string, error) {
	var name string
	err := wait.PollImmediate(Poll, 30*time.Second, func() (bool, error) {
		name = fmt.Sprintf("%v-%v", baseName, RandomSuffix())
		_, err := c.CoreV1().Namespaces().Get(context.TODO(), name, metav1.GetOptions{})
		if err == nil {
			// Already taken
			return false, nil
		}
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		Logf("Unexpected error while getting namespace: %v", err)
		return false, nil
	})
	return name, err
}

// CreateTestingNS should be used by every test, note that we append a common prefix to the provided test name.
// Please see NewFramework instead of using this directly.
func CreateTestingNS(baseName string, c clientset.Interface, labels map[string]string) (*v1.Namespace, error) {
	if labels == nil {
		labels = map[string]string{}
	}
	labels["e2e-run"] = string(RunID)

	// We don't use ObjectMeta.GenerateName feature, as in case of API call
	// failure we don't know whether the namespace was created and what is its
	// name.
	name, err := findAvailableNamespaceName(baseName, c)
	if err != nil {
		return nil, err
	}

	namespaceObj := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "",
			Labels:    labels,
		},
		Status: v1.NamespaceStatus{},
	}
	// Be robust about making the namespace creation call.
	var got *v1.Namespace
	if err := wait.PollImmediate(Poll, 30*time.Second, func() (bool, error) {
		var err error
		got, err = c.CoreV1().Namespaces().Create(context.TODO(), namespaceObj, metav1.CreateOptions{})
		if err != nil {
			Logf("Unexpected error while creating namespace: %v", err)
			return false, nil
		}
		return true, nil
	}); err != nil {
		return nil, err
	}

	if TestContext.VerifyServiceAccount {
		if err := WaitForDefaultServiceAccountInNamespace(c, got.Name); err != nil {
			// Even if we fail to create serviceAccount in the namespace,
			// we have successfully create a namespace.
			// So, return the created namespace.
			return got, err
		}
	}
	return got, nil
}

// CheckTestingNSDeletedExcept checks whether all e2e based existing namespaces are in the Terminating state
// and waits until they are finally deleted. It ignores namespace skip.
func CheckTestingNSDeletedExcept(c clientset.Interface, skip string) error {
	// TODO: Since we don't have support for bulk resource deletion in the API,
	// while deleting a namespace we are deleting all objects from that namespace
	// one by one (one deletion == one API call). This basically exposes us to
	// throttling - currently controller-manager has a limit of max 20 QPS.
	// Once #10217 is implemented and used in namespace-controller, deleting all
	// object from a given namespace should be much faster and we will be able
	// to lower this timeout.
	// However, now Density test is producing ~26000 events and Load capacity test
	// is producing ~35000 events, thus assuming there are no other requests it will
	// take ~30 minutes to fully delete the namespace. Thus I'm setting it to 60
	// minutes to avoid any timeouts here.
	timeout := 60 * time.Minute

	Logf("Waiting for terminating namespaces to be deleted...")
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(15 * time.Second) {
		namespaces, err := c.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			Logf("Listing namespaces failed: %v", err)
			continue
		}
		terminating := 0
		for _, ns := range namespaces.Items {
			if strings.HasPrefix(ns.ObjectMeta.Name, "e2e-tests-") && ns.ObjectMeta.Name != skip {
				if ns.Status.Phase == v1.NamespaceActive {
					return fmt.Errorf("Namespace %s is active", ns.ObjectMeta.Name)
				}
				terminating++
			}
		}
		if terminating == 0 {
			return nil
		}
	}
	return fmt.Errorf("Waiting for terminating namespaces to be deleted timed out")
}

// WaitForService waits until the service appears (exist == true), or disappears (exist == false)
func WaitForService(c clientset.Interface, namespace, name string, exist bool, interval, timeout time.Duration) error {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		_, err := c.CoreV1().Services(namespace).Get(context.TODO(), name, metav1.GetOptions{})
		switch {
		case err == nil:
			Logf("Service %s in namespace %s found.", name, namespace)
			return exist, nil
		case apierrors.IsNotFound(err):
			Logf("Service %s in namespace %s disappeared.", name, namespace)
			return !exist, nil
		case !testutils.IsRetryableAPIError(err):
			Logf("Non-retryable failure while getting service.")
			return false, err
		default:
			Logf("Get service %s in namespace %s failed: %v", name, namespace, err)
			return false, nil
		}
	})
	if err != nil {
		stateMsg := map[bool]string{true: "to appear", false: "to disappear"}
		return fmt.Errorf("error waiting for service %s/%s %s: %v", namespace, name, stateMsg[exist], err)
	}
	return nil
}

//WaitForServiceEndpointsNum waits until the amount of endpoints that implement service to expectNum.
func WaitForServiceEndpointsNum(c clientset.Interface, namespace, serviceName string, expectNum int, interval, timeout time.Duration) error {
	return wait.Poll(interval, timeout, func() (bool, error) {
		Logf("Waiting for amount of service:%s endpoints to be %d", serviceName, expectNum)
		list, err := c.CoreV1().Endpoints(namespace).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}

		for _, e := range list.Items {
			if e.Name == serviceName && countEndpointsNum(&e) == expectNum {
				return true, nil
			}
		}
		return false, nil
	})
}

func countEndpointsNum(e *v1.Endpoints) int {
	num := 0
	for _, sub := range e.Subsets {
		num += len(sub.Addresses)
	}
	return num
}

// restclientConfig returns a config holds the information needed to build connection to kubernetes clusters.
func restclientConfig(kubeContext string) (*clientcmdapi.Config, error) {
	Logf(">>> kubeConfig: %s", TestContext.KubeConfig)
	if TestContext.KubeConfig == "" {
		return nil, fmt.Errorf("KubeConfig must be specified to load client config")
	}
	c, err := clientcmd.LoadFromFile(TestContext.KubeConfig)
	if err != nil {
		return nil, fmt.Errorf("error loading KubeConfig: %v", err.Error())
	}
	if kubeContext != "" {
		Logf(">>> kubeContext: %s", kubeContext)
		c.CurrentContext = kubeContext
	}
	return c, nil
}

// ClientConfigGetter is a func that returns getter to return a config.
type ClientConfigGetter func() (*restclient.Config, error)

// LoadConfig returns a config for a rest client with the UserAgent set to include the current test name.
func LoadConfig() (config *restclient.Config, err error) {
	defer func() {
		if err == nil && config != nil {
			testDesc := ginkgo.CurrentGinkgoTestDescription()
			if len(testDesc.ComponentTexts) > 0 {
				componentTexts := strings.Join(testDesc.ComponentTexts, " ")
				config.UserAgent = fmt.Sprintf("%s -- %s", rest.DefaultKubernetesUserAgent(), componentTexts)
			}
		}
	}()

	if TestContext.NodeE2E {
		// This is a node e2e test, apply the node e2e configuration
		return &restclient.Config{Host: TestContext.Host}, nil
	}
	c, err := restclientConfig(TestContext.KubeContext)
	if err != nil {
		if TestContext.KubeConfig == "" {
			return restclient.InClusterConfig()
		}
		return nil, err
	}
	// In case Host is not set in TestContext, sets it as
	// CurrentContext Server for k8s API client to connect to.
	if TestContext.Host == "" && c.Clusters != nil {
		currentContext, ok := c.Clusters[c.CurrentContext]
		if ok {
			TestContext.Host = currentContext.Server
		}
	}

	return clientcmd.NewDefaultClientConfig(*c, &clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: TestContext.Host}}).ClientConfig()
}

// LoadClientset returns clientset for connecting to kubernetes clusters.
func LoadClientset() (*clientset.Clientset, error) {
	config, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err.Error())
	}
	return clientset.NewForConfig(config)
}

// RandomSuffix provides a random sequence to append to pods,services,rcs.
func RandomSuffix() string {
	return strconv.Itoa(rand.Intn(10000))
}

// Cleanup stops everything from filePath from namespace ns and checks if everything matching selectors from the given namespace is correctly stopped.
func Cleanup(filePath, ns string, selectors ...string) {
	ginkgo.By("using delete to clean up resources")
	var nsArg string
	if ns != "" {
		nsArg = fmt.Sprintf("--namespace=%s", ns)
	}
	RunKubectlOrDie(ns, "delete", "--grace-period=0", "-f", filePath, nsArg)
	AssertCleanup(ns, selectors...)
}

// AssertCleanup asserts that cleanup of a namespace wrt selectors occurred.
func AssertCleanup(ns string, selectors ...string) {
	var nsArg string
	if ns != "" {
		nsArg = fmt.Sprintf("--namespace=%s", ns)
	}

	var e error
	verifyCleanupFunc := func() (bool, error) {
		e = nil
		for _, selector := range selectors {
			resources := RunKubectlOrDie(ns, "get", "rc,svc", "-l", selector, "--no-headers", nsArg)
			if resources != "" {
				e = fmt.Errorf("Resources left running after stop:\n%s", resources)
				return false, nil
			}
			pods := RunKubectlOrDie(ns, "get", "pods", "-l", selector, nsArg, "-o", "go-template={{ range .items }}{{ if not .metadata.deletionTimestamp }}{{ .metadata.name }}{{ \"\\n\" }}{{ end }}{{ end }}")
			if pods != "" {
				e = fmt.Errorf("Pods left unterminated after stop:\n%s", pods)
				return false, nil
			}
		}
		return true, nil
	}
	err := wait.PollImmediate(500*time.Millisecond, 1*time.Minute, verifyCleanupFunc)
	if err != nil {
		Failf(e.Error())
	}
}

// LookForStringInPodExec looks for the given string in the output of a command
// executed in a specific pod container.
// TODO(alejandrox1): move to pod/ subpkg once kubectl methods are refactored.
func LookForStringInPodExec(ns, podName string, command []string, expectedString string, timeout time.Duration) (result string, err error) {
	return lookForString(expectedString, timeout, func() string {
		// use the first container
		args := []string{"exec", podName, fmt.Sprintf("--namespace=%v", ns), "--"}
		args = append(args, command...)
		return RunKubectlOrDie(ns, args...)
	})
}

// lookForString looks for the given string in the output of fn, repeatedly calling fn until
// the timeout is reached or the string is found. Returns last log and possibly
// error if the string was not found.
// TODO(alejandrox1): move to pod/ subpkg once kubectl methods are refactored.
func lookForString(expectedString string, timeout time.Duration, fn func() string) (result string, err error) {
	for t := time.Now(); time.Since(t) < timeout; time.Sleep(Poll) {
		result = fn()
		if strings.Contains(result, expectedString) {
			return
		}
	}
	err = fmt.Errorf("Failed to find \"%s\", last result: \"%s\"", expectedString, result)
	return
}

// KubectlBuilder is used to build, customize and execute a kubectl Command.
// Add more functions to customize the builder as needed.
type KubectlBuilder struct {
	cmd     *exec.Cmd
	timeout <-chan time.Time
}

// NewKubectlCommand returns a KubectlBuilder for running kubectl.
func NewKubectlCommand(namespace string, args ...string) *KubectlBuilder {
	b := new(KubectlBuilder)
	tk := e2ekubectl.NewTestKubeconfig(TestContext.CertDir, TestContext.Host, TestContext.KubeConfig, TestContext.KubeContext, TestContext.KubectlPath, namespace)
	b.cmd = tk.KubectlCmd(args...)
	return b
}

// WithEnv sets the given environment and returns itself.
func (b *KubectlBuilder) WithEnv(env []string) *KubectlBuilder {
	b.cmd.Env = env
	return b
}

// WithTimeout sets the given timeout and returns itself.
func (b *KubectlBuilder) WithTimeout(t <-chan time.Time) *KubectlBuilder {
	b.timeout = t
	return b
}

// WithStdinData sets the given data to stdin and returns itself.
func (b KubectlBuilder) WithStdinData(data string) *KubectlBuilder {
	b.cmd.Stdin = strings.NewReader(data)
	return &b
}

// WithStdinReader sets the given reader and returns itself.
func (b KubectlBuilder) WithStdinReader(reader io.Reader) *KubectlBuilder {
	b.cmd.Stdin = reader
	return &b
}

// ExecOrDie runs the kubectl executable or dies if error occurs.
func (b KubectlBuilder) ExecOrDie(namespace string) string {
	str, err := b.Exec()
	// In case of i/o timeout error, try talking to the apiserver again after 2s before dying.
	// Note that we're still dying after retrying so that we can get visibility to triage it further.
	if isTimeout(err) {
		Logf("Hit i/o timeout error, talking to the server 2s later to see if it's temporary.")
		time.Sleep(2 * time.Second)
		retryStr, retryErr := RunKubectl(namespace, "version")
		Logf("stdout: %q", retryStr)
		Logf("err: %v", retryErr)
	}
	ExpectNoError(err)
	return str
}

func isTimeout(err error) bool {
	switch err := err.(type) {
	case *url.Error:
		if err, ok := err.Err.(net.Error); ok && err.Timeout() {
			return true
		}
	case net.Error:
		if err.Timeout() {
			return true
		}
	}
	return false
}

// Exec runs the kubectl executable.
func (b KubectlBuilder) Exec() (string, error) {
	var stdout, stderr bytes.Buffer
	cmd := b.cmd
	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	Logf("Running '%s %s'", cmd.Path, strings.Join(cmd.Args[1:], " ")) // skip arg[0] as it is printed separately
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("error starting %v:\nCommand stdout:\n%v\nstderr:\n%v\nerror:\n%v", cmd, cmd.Stdout, cmd.Stderr, err)
	}
	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Wait()
	}()
	select {
	case err := <-errCh:
		if err != nil {
			var rc = 127
			if ee, ok := err.(*exec.ExitError); ok {
				rc = int(ee.Sys().(syscall.WaitStatus).ExitStatus())
				Logf("rc: %d", rc)
			}
			return "", uexec.CodeExitError{
				Err:  fmt.Errorf("error running %v:\nCommand stdout:\n%v\nstderr:\n%v\nerror:\n%v", cmd, cmd.Stdout, cmd.Stderr, err),
				Code: rc,
			}
		}
	case <-b.timeout:
		b.cmd.Process.Kill()
		return "", fmt.Errorf("timed out waiting for command %v:\nCommand stdout:\n%v\nstderr:\n%v", cmd, cmd.Stdout, cmd.Stderr)
	}
	Logf("stderr: %q", stderr.String())
	Logf("stdout: %q", stdout.String())
	return stdout.String(), nil
}

// RunKubectlOrDie is a convenience wrapper over kubectlBuilder
func RunKubectlOrDie(namespace string, args ...string) string {
	return NewKubectlCommand(namespace, args...).ExecOrDie(namespace)
}

// RunKubectl is a convenience wrapper over kubectlBuilder
func RunKubectl(namespace string, args ...string) (string, error) {
	return NewKubectlCommand(namespace, args...).Exec()
}

// RunKubectlOrDieInput is a convenience wrapper over kubectlBuilder that takes input to stdin
func RunKubectlOrDieInput(namespace string, data string, args ...string) string {
	return NewKubectlCommand(namespace, args...).WithStdinData(data).ExecOrDie(namespace)
}

// RunKubectlInput is a convenience wrapper over kubectlBuilder that takes input to stdin
func RunKubectlInput(namespace string, data string, args ...string) (string, error) {
	return NewKubectlCommand(namespace, args...).WithStdinData(data).Exec()
}

// RunKubemciWithKubeconfig is a convenience wrapper over RunKubemciCmd
func RunKubemciWithKubeconfig(args ...string) (string, error) {
	if TestContext.KubeConfig != "" {
		args = append(args, "--"+clientcmd.RecommendedConfigPathFlag+"="+TestContext.KubeConfig)
	}
	return RunKubemciCmd(args...)
}

// RunKubemciCmd is a convenience wrapper over kubectlBuilder to run kubemci.
// It assumes that kubemci exists in PATH.
func RunKubemciCmd(args ...string) (string, error) {
	// kubemci is assumed to be in PATH.
	kubemci := "kubemci"
	b := new(KubectlBuilder)
	args = append(args, "--gcp-project="+TestContext.CloudConfig.ProjectID)

	b.cmd = exec.Command(kubemci, args...)
	return b.Exec()
}

// StartCmdAndStreamOutput returns stdout and stderr after starting the given cmd.
func StartCmdAndStreamOutput(cmd *exec.Cmd) (stdout, stderr io.ReadCloser, err error) {
	stdout, err = cmd.StdoutPipe()
	if err != nil {
		return
	}
	stderr, err = cmd.StderrPipe()
	if err != nil {
		return
	}
	Logf("Asynchronously running '%s %s'", cmd.Path, strings.Join(cmd.Args, " "))
	err = cmd.Start()
	return
}

// TryKill is rough equivalent of ctrl+c for cleaning up processes. Intended to be run in defer.
func TryKill(cmd *exec.Cmd) {
	if err := cmd.Process.Kill(); err != nil {
		Logf("ERROR failed to kill command %v! The process may leak", cmd)
	}
}

// testContainerOutputMatcher runs the given pod in the given namespace and waits
// for all of the containers in the podSpec to move into the 'Success' status, and tests
// the specified container log against the given expected output using the given matcher.
func (f *Framework) testContainerOutputMatcher(scenarioName string,
	pod *v1.Pod,
	containerIndex int,
	expectedOutput []string,
	matcher func(string, ...interface{}) gomegatypes.GomegaMatcher) {
	ginkgo.By(fmt.Sprintf("Creating a pod to test %v", scenarioName))
	if containerIndex < 0 || containerIndex >= len(pod.Spec.Containers) {
		Failf("Invalid container index: %d", containerIndex)
	}
	ExpectNoError(f.MatchContainerOutput(pod, pod.Spec.Containers[containerIndex].Name, expectedOutput, matcher))
}

// MatchContainerOutput creates a pod and waits for all it's containers to exit with success.
// It then tests that the matcher with each expectedOutput matches the output of the specified container.
func (f *Framework) MatchContainerOutput(
	pod *v1.Pod,
	containerName string,
	expectedOutput []string,
	matcher func(string, ...interface{}) gomegatypes.GomegaMatcher) error {
	ns := pod.ObjectMeta.Namespace
	if ns == "" {
		ns = f.Namespace.Name
	}
	podClient := f.PodClientNS(ns)

	createdPod := podClient.Create(pod)
	defer func() {
		ginkgo.By("delete the pod")
		podClient.DeleteSync(createdPod.Name, metav1.DeleteOptions{}, DefaultPodDeletionTimeout)
	}()

	// Wait for client pod to complete.
	podErr := e2epod.WaitForPodSuccessInNamespace(f.ClientSet, createdPod.Name, ns)

	// Grab its logs.  Get host first.
	podStatus, err := podClient.Get(context.TODO(), createdPod.Name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get pod status: %v", err)
	}

	if podErr != nil {
		// Pod failed. Dump all logs from all containers to see what's wrong
		_ = podutil.VisitContainers(&podStatus.Spec, func(c *v1.Container) bool {
			logs, err := e2epod.GetPodLogs(f.ClientSet, ns, podStatus.Name, c.Name)
			if err != nil {
				Logf("Failed to get logs from node %q pod %q container %q: %v",
					podStatus.Spec.NodeName, podStatus.Name, c.Name, err)
			} else {
				Logf("Output of node %q pod %q container %q: %s", podStatus.Spec.NodeName, podStatus.Name, c.Name, logs)
			}
			return true
		})
		return fmt.Errorf("expected pod %q success: %v", createdPod.Name, podErr)
	}

	Logf("Trying to get logs from node %s pod %s container %s: %v",
		podStatus.Spec.NodeName, podStatus.Name, containerName, err)

	// Sometimes the actual containers take a second to get started, try to get logs for 60s
	logs, err := e2epod.GetPodLogs(f.ClientSet, ns, podStatus.Name, containerName)
	if err != nil {
		Logf("Failed to get logs from node %q pod %q container %q. %v",
			podStatus.Spec.NodeName, podStatus.Name, containerName, err)
		return fmt.Errorf("failed to get logs from %s for %s: %v", podStatus.Name, containerName, err)
	}

	for _, expected := range expectedOutput {
		m := matcher(expected)
		matches, err := m.Match(logs)
		if err != nil {
			return fmt.Errorf("expected %q in container output: %v", expected, err)
		} else if !matches {
			return fmt.Errorf("expected %q in container output: %s", expected, m.FailureMessage(logs))
		}
	}

	return nil
}

// EventsLister is a func that lists events.
type EventsLister func(opts metav1.ListOptions, ns string) (*v1.EventList, error)

// dumpEventsInNamespace dumps events in the given namespace.
func dumpEventsInNamespace(eventsLister EventsLister, namespace string) {
	ginkgo.By(fmt.Sprintf("Collecting events from namespace %q.", namespace))
	events, err := eventsLister(metav1.ListOptions{}, namespace)
	ExpectNoError(err, "failed to list events in namespace %q", namespace)

	ginkgo.By(fmt.Sprintf("Found %d events.", len(events.Items)))
	// Sort events by their first timestamp
	sortedEvents := events.Items
	if len(sortedEvents) > 1 {
		sort.Sort(byFirstTimestamp(sortedEvents))
	}
	for _, e := range sortedEvents {
		Logf("At %v - event for %v: %v %v: %v", e.FirstTimestamp, e.InvolvedObject.Name, e.Source, e.Reason, e.Message)
	}
	// Note that we don't wait for any Cleanup to propagate, which means
	// that if you delete a bunch of pods right before ending your test,
	// you may or may not see the killing/deletion/Cleanup events.
}

// DumpAllNamespaceInfo dumps events, pods and nodes information in the given namespace.
func DumpAllNamespaceInfo(c clientset.Interface, namespace string) {
	dumpEventsInNamespace(func(opts metav1.ListOptions, ns string) (*v1.EventList, error) {
		return c.CoreV1().Events(ns).List(context.TODO(), opts)
	}, namespace)

	e2epod.DumpAllPodInfoForNamespace(c, namespace)

	// If cluster is large, then the following logs are basically useless, because:
	// 1. it takes tens of minutes or hours to grab all of them
	// 2. there are so many of them that working with them are mostly impossible
	// So we dump them only if the cluster is relatively small.
	maxNodesForDump := TestContext.MaxNodesToGather
	nodes, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		Logf("unable to fetch node list: %v", err)
		return
	}
	if len(nodes.Items) <= maxNodesForDump {
		dumpAllNodeInfo(c, nodes)
	} else {
		Logf("skipping dumping cluster info - cluster too large")
	}
}

// byFirstTimestamp sorts a slice of events by first timestamp, using their involvedObject's name as a tie breaker.
type byFirstTimestamp []v1.Event

func (o byFirstTimestamp) Len() int      { return len(o) }
func (o byFirstTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o byFirstTimestamp) Less(i, j int) bool {
	if o[i].FirstTimestamp.Equal(&o[j].FirstTimestamp) {
		return o[i].InvolvedObject.Name < o[j].InvolvedObject.Name
	}
	return o[i].FirstTimestamp.Before(&o[j].FirstTimestamp)
}

func dumpAllNodeInfo(c clientset.Interface, nodes *v1.NodeList) {
	names := make([]string, len(nodes.Items))
	for ix := range nodes.Items {
		names[ix] = nodes.Items[ix].Name
	}
	DumpNodeDebugInfo(c, names, Logf)
}

// DumpNodeDebugInfo dumps debug information of the given nodes.
func DumpNodeDebugInfo(c clientset.Interface, nodeNames []string, logFunc func(fmt string, args ...interface{})) {
	for _, n := range nodeNames {
		logFunc("\nLogging node info for node %v", n)
		node, err := c.CoreV1().Nodes().Get(context.TODO(), n, metav1.GetOptions{})
		if err != nil {
			logFunc("Error getting node info %v", err)
		}
		logFunc("Node Info: %v", node)

		logFunc("\nLogging kubelet events for node %v", n)
		for _, e := range getNodeEvents(c, n) {
			logFunc("source %v type %v message %v reason %v first ts %v last ts %v, involved obj %+v",
				e.Source, e.Type, e.Message, e.Reason, e.FirstTimestamp, e.LastTimestamp, e.InvolvedObject)
		}
		logFunc("\nLogging pods the kubelet thinks is on node %v", n)
		podList, err := getKubeletPods(c, n)
		if err != nil {
			logFunc("Unable to retrieve kubelet pods for node %v: %v", n, err)
			continue
		}
		for _, p := range podList.Items {
			logFunc("%v started at %v (%d+%d container statuses recorded)", p.Name, p.Status.StartTime, len(p.Status.InitContainerStatuses), len(p.Status.ContainerStatuses))
			for _, c := range p.Status.InitContainerStatuses {
				logFunc("\tInit container %v ready: %v, restart count %v",
					c.Name, c.Ready, c.RestartCount)
			}
			for _, c := range p.Status.ContainerStatuses {
				logFunc("\tContainer %v ready: %v, restart count %v",
					c.Name, c.Ready, c.RestartCount)
			}
		}
		e2emetrics.HighLatencyKubeletOperations(c, 10*time.Second, n, logFunc)
		// TODO: Log node resource info
	}
}

// getKubeletPods retrieves the list of pods on the kubelet.
func getKubeletPods(c clientset.Interface, node string) (*v1.PodList, error) {
	var client restclient.Result
	finished := make(chan struct{}, 1)
	go func() {
		// call chain tends to hang in some cases when Node is not ready. Add an artificial timeout for this call. #22165
		client = c.CoreV1().RESTClient().Get().
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", node, ports.KubeletPort)).
			Suffix("pods").
			Do(context.TODO())

		finished <- struct{}{}
	}()
	select {
	case <-finished:
		result := &v1.PodList{}
		if err := client.Into(result); err != nil {
			return &v1.PodList{}, err
		}
		return result, nil
	case <-time.After(PodGetTimeout):
		return &v1.PodList{}, fmt.Errorf("Waiting up to %v for getting the list of pods", PodGetTimeout)
	}
}

// logNodeEvents logs kubelet events from the given node. This includes kubelet
// restart and node unhealthy events. Note that listing events like this will mess
// with latency metrics, beware of calling it during a test.
func getNodeEvents(c clientset.Interface, nodeName string) []v1.Event {
	selector := fields.Set{
		"involvedObject.kind":      "Node",
		"involvedObject.name":      nodeName,
		"involvedObject.namespace": metav1.NamespaceAll,
		"source":                   "kubelet",
	}.AsSelector().String()
	options := metav1.ListOptions{FieldSelector: selector}
	events, err := c.CoreV1().Events(metav1.NamespaceSystem).List(context.TODO(), options)
	if err != nil {
		Logf("Unexpected error retrieving node events %v", err)
		return []v1.Event{}
	}
	return events.Items
}

// WaitForAllNodesSchedulable waits up to timeout for all
// (but TestContext.AllowedNotReadyNodes) to become scheduable.
func WaitForAllNodesSchedulable(c clientset.Interface, timeout time.Duration) error {
	Logf("Waiting up to %v for all (but %d) nodes to be schedulable", timeout, TestContext.AllowedNotReadyNodes)

	return wait.PollImmediate(
		30*time.Second,
		timeout,
		e2enode.CheckReadyForTests(c, TestContext.NonblockingTaints, TestContext.AllowedNotReadyNodes, largeClusterThreshold),
	)
}

// GetPodSecretUpdateTimeout reuturns the timeout duration for updating pod secret.
func GetPodSecretUpdateTimeout(c clientset.Interface) time.Duration {
	// With SecretManager(ConfigMapManager), we may have to wait up to full sync period +
	// TTL of secret(configmap) to elapse before the Kubelet projects the update into the
	// volume and the container picks it up.
	// So this timeout is based on default Kubelet sync period (1 minute) + maximum TTL for
	// secret(configmap) that's based on cluster size + additional time as a fudge factor.
	secretTTL, err := getNodeTTLAnnotationValue(c)
	if err != nil {
		Logf("Couldn't get node TTL annotation (using default value of 0): %v", err)
	}
	podLogTimeout := 240*time.Second + secretTTL
	return podLogTimeout
}

func getNodeTTLAnnotationValue(c clientset.Interface) (time.Duration, error) {
	nodes, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil || len(nodes.Items) == 0 {
		return time.Duration(0), fmt.Errorf("Couldn't list any nodes to get TTL annotation: %v", err)
	}
	// Since TTL the kubelet is using is stored in node object, for the timeout
	// purpose we take it from the first node (all of them should be the same).
	node := &nodes.Items[0]
	if node.Annotations == nil {
		return time.Duration(0), fmt.Errorf("No annotations found on the node")
	}
	value, ok := node.Annotations[v1.ObjectTTLAnnotationKey]
	if !ok {
		return time.Duration(0), fmt.Errorf("No TTL annotation found on the node")
	}
	intValue, err := strconv.Atoi(value)
	if err != nil {
		return time.Duration(0), fmt.Errorf("Cannot convert TTL annotation from %#v to int", *node)
	}
	return time.Duration(intValue) * time.Second, nil
}

// AddOrUpdateLabelOnNode adds the given label key and value to the given node or updates value.
func AddOrUpdateLabelOnNode(c clientset.Interface, nodeName string, labelKey, labelValue string) {
	ExpectNoError(testutils.AddLabelsToNode(c, nodeName, map[string]string{labelKey: labelValue}))
}

// ExpectNodeHasLabel expects that the given node has the given label pair.
func ExpectNodeHasLabel(c clientset.Interface, nodeName string, labelKey string, labelValue string) {
	ginkgo.By("verifying the node has the label " + labelKey + " " + labelValue)
	node, err := c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	ExpectNoError(err)
	ExpectEqual(node.Labels[labelKey], labelValue)
}

// RemoveTaintOffNode removes the given taint from the given node.
func RemoveTaintOffNode(c clientset.Interface, nodeName string, taint v1.Taint) {
	ExpectNoError(controller.RemoveTaintOffNode(c, nodeName, nil, &taint))
	verifyThatTaintIsGone(c, nodeName, &taint)
}

// AddOrUpdateTaintOnNode adds the given taint to the given node or updates taint.
func AddOrUpdateTaintOnNode(c clientset.Interface, nodeName string, taint v1.Taint) {
	ExpectNoError(controller.AddOrUpdateTaintOnNode(c, nodeName, &taint))
}

// RemoveLabelOffNode is for cleaning up labels temporarily added to node,
// won't fail if target label doesn't exist or has been removed.
func RemoveLabelOffNode(c clientset.Interface, nodeName string, labelKey string) {
	ginkgo.By("removing the label " + labelKey + " off the node " + nodeName)
	ExpectNoError(testutils.RemoveLabelOffNode(c, nodeName, []string{labelKey}))

	ginkgo.By("verifying the node doesn't have the label " + labelKey)
	ExpectNoError(testutils.VerifyLabelsRemoved(c, nodeName, []string{labelKey}))
}

func verifyThatTaintIsGone(c clientset.Interface, nodeName string, taint *v1.Taint) {
	ginkgo.By("verifying the node doesn't have the taint " + taint.ToString())
	nodeUpdated, err := c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	ExpectNoError(err)
	if taintutils.TaintExists(nodeUpdated.Spec.Taints, taint) {
		Failf("Failed removing taint " + taint.ToString() + " of the node " + nodeName)
	}
}

// ExpectNodeHasTaint expects that the node has the given taint.
func ExpectNodeHasTaint(c clientset.Interface, nodeName string, taint *v1.Taint) {
	ginkgo.By("verifying the node has the taint " + taint.ToString())
	if has, err := NodeHasTaint(c, nodeName, taint); !has {
		ExpectNoError(err)
		Failf("Failed to find taint %s on node %s", taint.ToString(), nodeName)
	}
}

// NodeHasTaint returns true if the node has the given taint, else returns false.
func NodeHasTaint(c clientset.Interface, nodeName string, taint *v1.Taint) (bool, error) {
	node, err := c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	if err != nil {
		return false, err
	}

	nodeTaints := node.Spec.Taints

	if len(nodeTaints) == 0 || !taintutils.TaintExists(nodeTaints, taint) {
		return false, nil
	}
	return true, nil
}

// ScaleResource scales resource to the given size.
func ScaleResource(
	clientset clientset.Interface,
	scalesGetter scaleclient.ScalesGetter,
	ns, name string,
	size uint,
	wait bool,
	kind schema.GroupKind,
	gvr schema.GroupVersionResource,
) error {
	ginkgo.By(fmt.Sprintf("Scaling %v %s in namespace %s to %d", kind, name, ns, size))
	if err := testutils.ScaleResourceWithRetries(scalesGetter, ns, name, size, gvr); err != nil {
		return fmt.Errorf("error while scaling RC %s to %d replicas: %v", name, size, err)
	}
	if !wait {
		return nil
	}
	return e2epod.WaitForControlledPodsRunning(clientset, ns, name, kind)
}

// DeleteResourceAndWaitForGC deletes only given resource and waits for GC to delete the pods.
func DeleteResourceAndWaitForGC(c clientset.Interface, kind schema.GroupKind, ns, name string) error {
	ginkgo.By(fmt.Sprintf("deleting %v %s in namespace %s, will wait for the garbage collector to delete the pods", kind, name, ns))

	rtObject, err := e2eresource.GetRuntimeObjectForKind(c, kind, ns, name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			Logf("%v %s not found: %v", kind, name, err)
			return nil
		}
		return err
	}
	selector, err := e2eresource.GetSelectorFromRuntimeObject(rtObject)
	if err != nil {
		return err
	}
	replicas, err := e2eresource.GetReplicasFromRuntimeObject(rtObject)
	if err != nil {
		return err
	}

	ps, err := testutils.NewPodStore(c, ns, selector, fields.Everything())
	if err != nil {
		return err
	}

	defer ps.Stop()
	falseVar := false
	deleteOption := metav1.DeleteOptions{OrphanDependents: &falseVar}
	startTime := time.Now()
	if err := testutils.DeleteResourceWithRetries(c, kind, ns, name, deleteOption); err != nil {
		return err
	}
	deleteTime := time.Since(startTime)
	Logf("Deleting %v %s took: %v", kind, name, deleteTime)

	var interval, timeout time.Duration
	switch {
	case replicas < 100:
		interval = 100 * time.Millisecond
	case replicas < 1000:
		interval = 1 * time.Second
	default:
		interval = 10 * time.Second
	}
	if replicas < 5000 {
		timeout = 10 * time.Minute
	} else {
		timeout = time.Duration(replicas/gcThroughput) * time.Second
		// gcThroughput is pretty strict now, add a bit more to it
		timeout = timeout + 3*time.Minute
	}

	err = waitForPodsInactive(ps, interval, timeout)
	if err != nil {
		return fmt.Errorf("error while waiting for pods to become inactive %s: %v", name, err)
	}
	terminatePodTime := time.Since(startTime) - deleteTime
	Logf("Terminating %v %s pods took: %v", kind, name, terminatePodTime)

	// In gce, at any point, small percentage of nodes can disappear for
	// ~10 minutes due to hostError. 20 minutes should be long enough to
	// restart VM in that case and delete the pod.
	err = waitForPodsGone(ps, interval, 20*time.Minute)
	if err != nil {
		return fmt.Errorf("error while waiting for pods gone %s: %v", name, err)
	}
	return nil
}

// waitForPodsGone waits until there are no pods left in the PodStore.
func waitForPodsGone(ps *testutils.PodStore, interval, timeout time.Duration) error {
	var pods []*v1.Pod
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		if pods = ps.List(); len(pods) == 0 {
			return true, nil
		}
		return false, nil
	})

	if err == wait.ErrWaitTimeout {
		for _, pod := range pods {
			Logf("ERROR: Pod %q still exists. Node: %q", pod.Name, pod.Spec.NodeName)
		}
		return fmt.Errorf("there are %d pods left. E.g. %q on node %q", len(pods), pods[0].Name, pods[0].Spec.NodeName)
	}
	return err
}

// waitForPodsInactive waits until there are no active pods left in the PodStore.
// This is to make a fair comparison of deletion time between DeleteRCAndPods
// and DeleteRCAndWaitForGC, because the RC controller decreases status.replicas
// when the pod is inactvie.
func waitForPodsInactive(ps *testutils.PodStore, interval, timeout time.Duration) error {
	var activePods []*v1.Pod
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		pods := ps.List()
		activePods = controller.FilterActivePods(pods)
		if len(activePods) != 0 {
			return false, nil
		}
		return true, nil
	})

	if err == wait.ErrWaitTimeout {
		for _, pod := range activePods {
			Logf("ERROR: Pod %q running on %q is still active", pod.Name, pod.Spec.NodeName)
		}
		return fmt.Errorf("there are %d active pods. E.g. %q on node %q", len(activePods), activePods[0].Name, activePods[0].Spec.NodeName)
	}
	return err
}

// RunHostCmd runs the given cmd in the context of the given pod using `kubectl exec`
// inside of a shell.
func RunHostCmd(ns, name, cmd string) (string, error) {
	return RunKubectl(ns, "exec", fmt.Sprintf("--namespace=%v", ns), name, "--", "/bin/sh", "-x", "-c", cmd)
}

// RunHostCmdOrDie calls RunHostCmd and dies on error.
func RunHostCmdOrDie(ns, name, cmd string) string {
	stdout, err := RunHostCmd(ns, name, cmd)
	Logf("stdout: %v", stdout)
	ExpectNoError(err)
	return stdout
}

// RunHostCmdWithRetries calls RunHostCmd and retries all errors
// until it succeeds or the specified timeout expires.
// This can be used with idempotent commands to deflake transient Node issues.
func RunHostCmdWithRetries(ns, name, cmd string, interval, timeout time.Duration) (string, error) {
	start := time.Now()
	for {
		out, err := RunHostCmd(ns, name, cmd)
		if err == nil {
			return out, nil
		}
		if elapsed := time.Since(start); elapsed > timeout {
			return out, fmt.Errorf("RunHostCmd still failed after %v: %v", elapsed, err)
		}
		Logf("Waiting %v to retry failed RunHostCmd: %v", interval, err)
		time.Sleep(interval)
	}
}

// AllNodesReady checks whether all registered nodes are ready.
// TODO: we should change the AllNodesReady call in AfterEach to WaitForAllNodesHealthy,
// and figure out how to do it in a configurable way, as we can't expect all setups to run
// default test add-ons.
func AllNodesReady(c clientset.Interface, timeout time.Duration) error {
	Logf("Waiting up to %v for all (but %d) nodes to be ready", timeout, TestContext.AllowedNotReadyNodes)

	var notReady []*v1.Node
	err := wait.PollImmediate(Poll, timeout, func() (bool, error) {
		notReady = nil
		// It should be OK to list unschedulable Nodes here.
		nodes, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		for i := range nodes.Items {
			node := &nodes.Items[i]
			if !e2enode.IsConditionSetAsExpected(node, v1.NodeReady, true) {
				notReady = append(notReady, node)
			}
		}
		// Framework allows for <TestContext.AllowedNotReadyNodes> nodes to be non-ready,
		// to make it possible e.g. for incorrect deployment of some small percentage
		// of nodes (which we allow in cluster validation). Some nodes that are not
		// provisioned correctly at startup will never become ready (e.g. when something
		// won't install correctly), so we can't expect them to be ready at any point.
		return len(notReady) <= TestContext.AllowedNotReadyNodes, nil
	})

	if err != nil && err != wait.ErrWaitTimeout {
		return err
	}

	if len(notReady) > TestContext.AllowedNotReadyNodes {
		msg := ""
		for _, node := range notReady {
			msg = fmt.Sprintf("%s, %s", msg, node.Name)
		}
		return fmt.Errorf("Not ready nodes: %#v", msg)
	}
	return nil
}

// RestartKubelet restarts kubelet on the given host.
func RestartKubelet(host string) error {
	// TODO: Make it work for all providers and distros.
	supportedProviders := []string{"gce", "aws", "vsphere"}
	if !ProviderIs(supportedProviders...) {
		return fmt.Errorf("unsupported provider for RestartKubelet: %s, supported providers are: %v", TestContext.Provider, supportedProviders)
	}
	if ProviderIs("gce") && !NodeOSDistroIs("debian", "gci") {
		return fmt.Errorf("unsupported node OS distro: %s", TestContext.NodeOSDistro)
	}
	var cmd string

	if ProviderIs("gce") && NodeOSDistroIs("debian") {
		cmd = "sudo /etc/init.d/kubelet restart"
	} else if ProviderIs("vsphere") {
		var sudoPresent bool
		sshResult, err := e2essh.SSH("sudo --version", host, TestContext.Provider)
		if err != nil {
			return fmt.Errorf("Unable to ssh to host %s with error %v", host, err)
		}
		if !strings.Contains(sshResult.Stderr, "command not found") {
			sudoPresent = true
		}
		sshResult, err = e2essh.SSH("systemctl --version", host, TestContext.Provider)
		if err != nil {
			return fmt.Errorf("Failed to execute command 'systemctl' on host %s with error %v", host, err)
		}
		if !strings.Contains(sshResult.Stderr, "command not found") {
			cmd = "systemctl restart kubelet"
		} else {
			cmd = "service kubelet restart"
		}
		if sudoPresent {
			cmd = fmt.Sprintf("sudo %s", cmd)
		}
	} else {
		cmd = "sudo systemctl restart kubelet"
	}
	Logf("Restarting kubelet via ssh on host %s with command %s", host, cmd)
	result, err := e2essh.SSH(cmd, host, TestContext.Provider)
	if err != nil || result.Code != 0 {
		e2essh.LogResult(result)
		return fmt.Errorf("couldn't restart kubelet: %v", err)
	}
	return nil
}

// RestartApiserver restarts the kube-apiserver.
func RestartApiserver(namespace string, cs clientset.Interface) error {
	// TODO: Make it work for all providers.
	if !ProviderIs("gce", "gke", "aws") {
		return fmt.Errorf("unsupported provider for RestartApiserver: %s", TestContext.Provider)
	}
	if ProviderIs("gce", "aws") {
		initialRestartCount, err := getApiserverRestartCount(cs)
		if err != nil {
			return fmt.Errorf("failed to get apiserver's restart count: %v", err)
		}
		if err := sshRestartMaster(); err != nil {
			return fmt.Errorf("failed to restart apiserver: %v", err)
		}
		return waitForApiserverRestarted(cs, initialRestartCount)
	}
	// GKE doesn't allow ssh access, so use a same-version master
	// upgrade to teardown/recreate master.
	v, err := cs.Discovery().ServerVersion()
	if err != nil {
		return err
	}
	return masterUpgradeGKE(namespace, v.GitVersion[1:]) // strip leading 'v'
}

func sshRestartMaster() error {
	if !ProviderIs("gce", "aws") {
		return fmt.Errorf("unsupported provider for sshRestartMaster: %s", TestContext.Provider)
	}
	var command string
	if ProviderIs("gce") {
		command = "pidof kube-apiserver | xargs sudo kill"
	} else {
		command = "sudo /etc/init.d/kube-apiserver restart"
	}
	Logf("Restarting master via ssh, running: %v", command)
	result, err := e2essh.SSH(command, net.JoinHostPort(GetMasterHost(), sshPort), TestContext.Provider)
	if err != nil || result.Code != 0 {
		e2essh.LogResult(result)
		return fmt.Errorf("couldn't restart apiserver: %v", err)
	}
	return nil
}

// waitForApiserverRestarted waits until apiserver's restart count increased.
func waitForApiserverRestarted(c clientset.Interface, initialRestartCount int32) error {
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		restartCount, err := getApiserverRestartCount(c)
		if err != nil {
			Logf("Failed to get apiserver's restart count: %v", err)
			continue
		}
		if restartCount > initialRestartCount {
			Logf("Apiserver has restarted.")
			return nil
		}
		Logf("Waiting for apiserver restart count to increase")
	}
	return fmt.Errorf("timed out waiting for apiserver to be restarted")
}

func getApiserverRestartCount(c clientset.Interface) (int32, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"component": "kube-apiserver"}))
	listOpts := metav1.ListOptions{LabelSelector: label.String()}
	pods, err := c.CoreV1().Pods(metav1.NamespaceSystem).List(context.TODO(), listOpts)
	if err != nil {
		return -1, err
	}
	if len(pods.Items) != 1 {
		return -1, fmt.Errorf("unexpected number of apiserver pod: %d", len(pods.Items))
	}
	for _, s := range pods.Items[0].Status.ContainerStatuses {
		if s.Name != "kube-apiserver" {
			continue
		}
		return s.RestartCount, nil
	}
	return -1, fmt.Errorf("Failed to find kube-apiserver container in pod")
}

// RestartControllerManager restarts the kube-controller-manager.
func RestartControllerManager() error {
	// TODO: Make it work for all providers and distros.
	if !ProviderIs("gce", "aws") {
		return fmt.Errorf("unsupported provider for RestartControllerManager: %s", TestContext.Provider)
	}
	if ProviderIs("gce") && !MasterOSDistroIs("gci") {
		return fmt.Errorf("unsupported master OS distro: %s", TestContext.MasterOSDistro)
	}
	cmd := "pidof kube-controller-manager | xargs sudo kill"
	Logf("Restarting controller-manager via ssh, running: %v", cmd)
	result, err := e2essh.SSH(cmd, net.JoinHostPort(GetMasterHost(), sshPort), TestContext.Provider)
	if err != nil || result.Code != 0 {
		e2essh.LogResult(result)
		return fmt.Errorf("couldn't restart controller-manager: %v", err)
	}
	return nil
}

// WaitForControllerManagerUp waits for the kube-controller-manager to be up.
func WaitForControllerManagerUp() error {
	cmd := "curl http://localhost:" + strconv.Itoa(ports.InsecureKubeControllerManagerPort) + "/healthz"
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		result, err := e2essh.SSH(cmd, net.JoinHostPort(GetMasterHost(), sshPort), TestContext.Provider)
		if err != nil || result.Code != 0 {
			e2essh.LogResult(result)
		}
		if result.Stdout == "ok" {
			return nil
		}
	}
	return fmt.Errorf("waiting for controller-manager timed out")
}

type extractRT struct {
	http.Header
}

func (rt *extractRT) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.Header = req.Header
	return &http.Response{}, nil
}

// headersForConfig extracts any http client logic necessary for the provided
// config.
func headersForConfig(c *restclient.Config, url *url.URL) (http.Header, error) {
	extract := &extractRT{}
	rt, err := restclient.HTTPWrappersForConfig(c, extract)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequest("GET", url.String(), nil)
	if err != nil {
		return nil, err
	}
	if _, err := rt.RoundTrip(request); err != nil {
		return nil, err
	}
	return extract.Header, nil
}

// OpenWebSocketForURL constructs a websocket connection to the provided URL, using the client
// config, with the specified protocols.
func OpenWebSocketForURL(url *url.URL, config *restclient.Config, protocols []string) (*websocket.Conn, error) {
	tlsConfig, err := restclient.TLSConfigFor(config)
	if err != nil {
		return nil, fmt.Errorf("Failed to create tls config: %v", err)
	}
	if url.Scheme == "https" {
		url.Scheme = "wss"
	} else {
		url.Scheme = "ws"
	}
	headers, err := headersForConfig(config, url)
	if err != nil {
		return nil, fmt.Errorf("Failed to load http headers: %v", err)
	}
	cfg, err := websocket.NewConfig(url.String(), "http://localhost")
	if err != nil {
		return nil, fmt.Errorf("Failed to create websocket config: %v", err)
	}
	cfg.Header = headers
	cfg.TlsConfig = tlsConfig
	cfg.Protocol = protocols
	return websocket.DialConfig(cfg)
}

// LookForStringInLog looks for the given string in the log of a specific pod container
func LookForStringInLog(ns, podName, container, expectedString string, timeout time.Duration) (result string, err error) {
	return lookForString(expectedString, timeout, func() string {
		return RunKubectlOrDie(ns, "logs", podName, container, fmt.Sprintf("--namespace=%v", ns))
	})
}

// EnsureLoadBalancerResourcesDeleted ensures that cloud load balancer resources that were created
// are actually cleaned up.  Currently only implemented for GCE/GKE.
func EnsureLoadBalancerResourcesDeleted(ip, portRange string) error {
	return TestContext.CloudConfig.Provider.EnsureLoadBalancerResourcesDeleted(ip, portRange)
}

// BlockNetwork blocks network between the given from value and the given to value.
// The following helper functions can block/unblock network from source
// host to destination host by manipulating iptable rules.
// This function assumes it can ssh to the source host.
//
// Caution:
// Recommend to input IP instead of hostnames. Using hostnames will cause iptables to
// do a DNS lookup to resolve the name to an IP address, which will
// slow down the test and cause it to fail if DNS is absent or broken.
//
// Suggested usage pattern:
// func foo() {
//	...
//	defer UnblockNetwork(from, to)
//	BlockNetwork(from, to)
//	...
// }
//
func BlockNetwork(from string, to string) {
	Logf("block network traffic from %s to %s", from, to)
	iptablesRule := fmt.Sprintf("OUTPUT --destination %s --jump REJECT", to)
	dropCmd := fmt.Sprintf("sudo iptables --insert %s", iptablesRule)
	if result, err := e2essh.SSH(dropCmd, from, TestContext.Provider); result.Code != 0 || err != nil {
		e2essh.LogResult(result)
		Failf("Unexpected error: %v", err)
	}
}

// UnblockNetwork unblocks network between the given from value and the given to value.
func UnblockNetwork(from string, to string) {
	Logf("Unblock network traffic from %s to %s", from, to)
	iptablesRule := fmt.Sprintf("OUTPUT --destination %s --jump REJECT", to)
	undropCmd := fmt.Sprintf("sudo iptables --delete %s", iptablesRule)
	// Undrop command may fail if the rule has never been created.
	// In such case we just lose 30 seconds, but the cluster is healthy.
	// But if the rule had been created and removing it failed, the node is broken and
	// not coming back. Subsequent tests will run or fewer nodes (some of the tests
	// may fail). Manual intervention is required in such case (recreating the
	// cluster solves the problem too).
	err := wait.Poll(time.Millisecond*100, time.Second*30, func() (bool, error) {
		result, err := e2essh.SSH(undropCmd, from, TestContext.Provider)
		if result.Code == 0 && err == nil {
			return true, nil
		}
		e2essh.LogResult(result)
		if err != nil {
			Logf("Unexpected error: %v", err)
		}
		return false, nil
	})
	if err != nil {
		Failf("Failed to remove the iptable REJECT rule. Manual intervention is "+
			"required on host %s: remove rule %s, if exists", from, iptablesRule)
	}
}

// CoreDump SSHs to the master and all nodes and dumps their logs into dir.
// It shells out to cluster/log-dump/log-dump.sh to accomplish this.
func CoreDump(dir string) {
	if TestContext.DisableLogDump {
		Logf("Skipping dumping logs from cluster")
		return
	}
	var cmd *exec.Cmd
	if TestContext.LogexporterGCSPath != "" {
		Logf("Dumping logs from nodes to GCS directly at path: %s", TestContext.LogexporterGCSPath)
		cmd = exec.Command(path.Join(TestContext.RepoRoot, "cluster", "log-dump", "log-dump.sh"), dir, TestContext.LogexporterGCSPath)
	} else {
		Logf("Dumping logs locally to: %s", dir)
		cmd = exec.Command(path.Join(TestContext.RepoRoot, "cluster", "log-dump", "log-dump.sh"), dir)
	}
	cmd.Env = append(os.Environ(), fmt.Sprintf("LOG_DUMP_SYSTEMD_SERVICES=%s", parseSystemdServices(TestContext.SystemdServices)))
	cmd.Env = append(os.Environ(), fmt.Sprintf("LOG_DUMP_SYSTEMD_JOURNAL=%v", TestContext.DumpSystemdJournal))

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		Logf("Error running cluster/log-dump/log-dump.sh: %v", err)
	}
}

// parseSystemdServices converts services separator from comma to space.
func parseSystemdServices(services string) string {
	return strings.TrimSpace(strings.Replace(services, ",", " ", -1))
}

// RunCmd runs cmd using args and returns its stdout and stderr. It also outputs
// cmd's stdout and stderr to their respective OS streams.
func RunCmd(command string, args ...string) (string, string, error) {
	return RunCmdEnv(nil, command, args...)
}

// RunCmdEnv runs cmd with the provided environment and args and
// returns its stdout and stderr. It also outputs cmd's stdout and
// stderr to their respective OS streams.
func RunCmdEnv(env []string, command string, args ...string) (string, string, error) {
	Logf("Running %s %v", command, args)
	var bout, berr bytes.Buffer
	cmd := exec.Command(command, args...)
	// We also output to the OS stdout/stderr to aid in debugging in case cmd
	// hangs and never returns before the test gets killed.
	//
	// This creates some ugly output because gcloud doesn't always provide
	// newlines.
	cmd.Stdout = io.MultiWriter(os.Stdout, &bout)
	cmd.Stderr = io.MultiWriter(os.Stderr, &berr)
	cmd.Env = env
	err := cmd.Run()
	stdout, stderr := bout.String(), berr.String()
	if err != nil {
		return "", "", fmt.Errorf("error running %s %v; got error %v, stdout %q, stderr %q",
			command, args, err, stdout, stderr)
	}
	return stdout, stderr, nil
}

// E2ETestNodePreparer implements testutils.TestNodePreparer interface, which is used
// to create/modify Nodes before running a test.
type E2ETestNodePreparer struct {
	client clientset.Interface
	// Specifies how many nodes should be modified using the given strategy.
	// Only one strategy can be applied to a single Node, so there needs to
	// be at least <sum_of_keys> Nodes in the cluster.
	countToStrategy       []testutils.CountToStrategy
	nodeToAppliedStrategy map[string]testutils.PrepareNodeStrategy
}

// PrepareNodes prepares nodes in the cluster.
func (p *E2ETestNodePreparer) PrepareNodes() error {
	nodes, err := e2enode.GetReadySchedulableNodes(p.client)
	if err != nil {
		return err
	}
	numTemplates := 0
	for _, v := range p.countToStrategy {
		numTemplates += v.Count
	}
	if numTemplates > len(nodes.Items) {
		return fmt.Errorf("Can't prepare Nodes. Got more templates than existing Nodes")
	}
	index := 0
	sum := 0
	for _, v := range p.countToStrategy {
		sum += v.Count
		for ; index < sum; index++ {
			if err := testutils.DoPrepareNode(p.client, &nodes.Items[index], v.Strategy); err != nil {
				klog.Errorf("Aborting node preparation: %v", err)
				return err
			}
			p.nodeToAppliedStrategy[nodes.Items[index].Name] = v.Strategy
		}
	}
	return nil
}

// CleanupNodes cleanups nodes in the cluster.
func (p *E2ETestNodePreparer) CleanupNodes() error {
	var encounteredError error
	nodes, err := e2enode.GetReadySchedulableNodes(p.client)
	if err != nil {
		return err
	}
	for i := range nodes.Items {
		name := nodes.Items[i].Name
		strategy, found := p.nodeToAppliedStrategy[name]
		if found {
			if err = testutils.DoCleanupNode(p.client, name, strategy); err != nil {
				klog.Errorf("Skipping cleanup of Node: failed update of %v: %v", name, err)
				encounteredError = err
			}
		}
	}
	return encounteredError
}

// getMasterAddresses returns the externalIP, internalIP and hostname fields of the master.
// If any of these is unavailable, it is set to "".
func getMasterAddresses(c clientset.Interface) (string, string, string) {
	var externalIP, internalIP, hostname string

	// Populate the internal IP.
	eps, err := c.CoreV1().Endpoints(metav1.NamespaceDefault).Get(context.TODO(), "kubernetes", metav1.GetOptions{})
	if err != nil {
		Failf("Failed to get kubernetes endpoints: %v", err)
	}
	if len(eps.Subsets) != 1 || len(eps.Subsets[0].Addresses) != 1 {
		Failf("There are more than 1 endpoints for kubernetes service: %+v", eps)
	}
	internalIP = eps.Subsets[0].Addresses[0].IP

	// Populate the external IP/hostname.
	hostURL, err := url.Parse(TestContext.Host)
	if err != nil {
		Failf("Failed to parse hostname: %v", err)
	}
	if net.ParseIP(hostURL.Host) != nil {
		externalIP = hostURL.Host
	} else {
		hostname = hostURL.Host
	}

	return externalIP, internalIP, hostname
}

// GetAllMasterAddresses returns all IP addresses on which the kubelet can reach the master.
// It may return internal and external IPs, even if we expect for
// e.g. internal IPs to be used (issue #56787), so that we can be
// sure to block the master fully during tests.
func GetAllMasterAddresses(c clientset.Interface) []string {
	externalIP, internalIP, _ := getMasterAddresses(c)

	ips := sets.NewString()
	switch TestContext.Provider {
	case "gce", "gke":
		if externalIP != "" {
			ips.Insert(externalIP)
		}
		if internalIP != "" {
			ips.Insert(internalIP)
		}
	case "aws":
		ips.Insert(awsMasterIP)
	default:
		Failf("This test is not supported for provider %s and should be disabled", TestContext.Provider)
	}
	return ips.List()
}

// DescribeIng describes information of ingress by running kubectl describe ing.
func DescribeIng(ns string) {
	Logf("\nOutput of kubectl describe ing:\n")
	desc, _ := RunKubectl(
		ns, "describe", "ing", fmt.Sprintf("--namespace=%v", ns))
	Logf(desc)
}

// NewTestPod returns a pod that has the specified requests and limits
func (f *Framework) NewTestPod(name string, requests v1.ResourceList, limits v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "pause",
					Image: imageutils.GetPauseImageName(),
					Resources: v1.ResourceRequirements{
						Requests: requests,
						Limits:   limits,
					},
				},
			},
		},
	}
}

// NewAgnhostPod returns a pod that uses the agnhost image. The image's binary supports various subcommands
// that behave the same, no matter the underlying OS.
func (f *Framework) NewAgnhostPod(name string, args ...string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "agnhost",
					Image: AgnHostImage,
					Args:  args,
				},
			},
		},
	}
}

// CreateEmptyFileOnPod creates empty file at given path on the pod.
// TODO(alejandrox1): move to subpkg pod once kubectl methods have been refactored.
func CreateEmptyFileOnPod(namespace string, podName string, filePath string) error {
	_, err := RunKubectl(namespace, "exec", fmt.Sprintf("--namespace=%s", namespace), podName, "--", "/bin/sh", "-c", fmt.Sprintf("touch %s", filePath))
	return err
}

// DumpDebugInfo dumps debug info of tests.
func DumpDebugInfo(c clientset.Interface, ns string) {
	sl, _ := c.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{LabelSelector: labels.Everything().String()})
	for _, s := range sl.Items {
		desc, _ := RunKubectl(ns, "describe", "po", s.Name, fmt.Sprintf("--namespace=%v", ns))
		Logf("\nOutput of kubectl describe %v:\n%v", s.Name, desc)

		l, _ := RunKubectl(ns, "logs", s.Name, fmt.Sprintf("--namespace=%v", ns), "--tail=100")
		Logf("\nLast 100 log lines of %v:\n%v", s.Name, l)
	}
}

// DsFromManifest reads a .json/yaml file and returns the daemonset in it.
func DsFromManifest(url string) (*appsv1.DaemonSet, error) {
	Logf("Parsing ds from %v", url)

	var response *http.Response
	var err error

	for i := 1; i <= 5; i++ {
		response, err = http.Get(url)
		if err == nil && response.StatusCode == 200 {
			break
		}
		time.Sleep(time.Duration(i) * time.Second)
	}

	if err != nil {
		return nil, fmt.Errorf("Failed to get url: %v", err)
	}
	if response.StatusCode != 200 {
		return nil, fmt.Errorf("invalid http response status: %v", response.StatusCode)
	}
	defer response.Body.Close()

	data, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("Failed to read html response body: %v", err)
	}
	return DsFromData(data)
}

// DsFromData reads a byte slice and returns the daemonset in it.
func DsFromData(data []byte) (*appsv1.DaemonSet, error) {
	var ds appsv1.DaemonSet
	dataJSON, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, fmt.Errorf("Failed to parse data to json: %v", err)
	}

	err = runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), dataJSON, &ds)
	if err != nil {
		return nil, fmt.Errorf("Failed to decode DaemonSet spec: %v", err)
	}
	return &ds, nil
}

// GetClusterZones returns the values of zone label collected from all nodes.
func GetClusterZones(c clientset.Interface) (sets.String, error) {
	nodes, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("Error getting nodes while attempting to list cluster zones: %v", err)
	}

	// collect values of zone label from all nodes
	zones := sets.NewString()
	for _, node := range nodes.Items {
		if zone, found := node.Labels[v1.LabelZoneFailureDomain]; found {
			zones.Insert(zone)
		}

		if zone, found := node.Labels[v1.LabelZoneFailureDomainStable]; found {
			zones.Insert(zone)
		}
	}
	return zones, nil
}

// GetFileModeRegex returns a file mode related regex which should be matched by the mounttest pods' output.
// If the given mask is nil, then the regex will contain the default OS file modes, which are 0644 for Linux and 0775 for Windows.
func GetFileModeRegex(filePath string, mask *int32) string {
	var (
		linuxMask   int32
		windowsMask int32
	)
	if mask == nil {
		linuxMask = int32(0644)
		windowsMask = int32(0775)
	} else {
		linuxMask = *mask
		windowsMask = *mask
	}

	linuxOutput := fmt.Sprintf("mode of file \"%s\": %v", filePath, os.FileMode(linuxMask))
	windowsOutput := fmt.Sprintf("mode of Windows file \"%v\": %s", filePath, os.FileMode(windowsMask))

	return fmt.Sprintf("(%s|%s)", linuxOutput, windowsOutput)
}

// PrettyPrintJSON converts metrics to JSON format.
func PrettyPrintJSON(metrics interface{}) string {
	output := &bytes.Buffer{}
	if err := json.NewEncoder(output).Encode(metrics); err != nil {
		Logf("Error building encoder: %v", err)
		return ""
	}
	formatted := &bytes.Buffer{}
	if err := json.Indent(formatted, output.Bytes(), "", "  "); err != nil {
		Logf("Error indenting: %v", err)
		return ""
	}
	return string(formatted.Bytes())
}
