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
	"math/rand"
	"net/url"
	"os"
	"os/exec"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	watchtools "k8s.io/client-go/tools/watch"
	netutils "k8s.io/utils/net"
)

// DEPRECATED constants. Use the timeouts in framework.Framework instead.
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

	// SnapshotDeleteTimeout is how long for snapshot to delete snapshotContent.
	SnapshotDeleteTimeout = 5 * time.Minute

	// ControlPlaneLabel is valid label for kubeadm based clusters like kops ONLY
	ControlPlaneLabel = "node-role.kubernetes.io/control-plane"
)

var (
	// ProvidersWithSSH are those providers where each node is accessible with SSH
	ProvidersWithSSH = []string{"gce", "aws", "local", "azure"}
)

// RunID is a unique identifier of the e2e run.
// Beware that this ID is not the same for all tests in the e2e run, because each Ginkgo node creates it separately.
var RunID = uuid.NewUUID()

// CreateTestingNSFn is a func that is responsible for creating namespace used for executing e2e tests.
type CreateTestingNSFn func(ctx context.Context, baseName string, c clientset.Interface, labels map[string]string) (*v1.Namespace, error)

// APIAddress returns a address of an instance.
func APIAddress() string {
	instanceURL, err := url.Parse(TestContext.Host)
	ExpectNoError(err)
	return instanceURL.Hostname()
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

// NodeOSArchIs returns true if the node OS arch is included in the supportedNodeOsArchs. Otherwise false.
func NodeOSArchIs(supportedNodeOsArchs ...string) bool {
	for _, arch := range supportedNodeOsArchs {
		if strings.EqualFold(arch, TestContext.NodeOSArch) {
			return true
		}
	}
	return false
}

// DeleteNamespaces deletes all namespaces that match the given delete and skip filters.
// Filter is by simple strings.Contains; first skip filter, then delete filter.
// Returns the list of deleted namespaces or an error.
func DeleteNamespaces(ctx context.Context, c clientset.Interface, deleteFilter, skipFilter []string) ([]string, error) {
	ginkgo.By("Deleting namespaces")
	nsList, err := c.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	ExpectNoError(err, "Failed to get namespace list")
	var deleted []string
	var wg sync.WaitGroup
OUTER:
	for _, item := range nsList.Items {
		for _, pattern := range skipFilter {
			if strings.Contains(item.Name, pattern) {
				continue OUTER
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
			gomega.Expect(c.CoreV1().Namespaces().Delete(ctx, nsName, metav1.DeleteOptions{})).To(gomega.Succeed())
			Logf("namespace : %v api call to delete is complete ", nsName)
		}(item.Name)
	}
	wg.Wait()
	return deleted, nil
}

// WaitForNamespacesDeleted waits for the namespaces to be deleted.
func WaitForNamespacesDeleted(ctx context.Context, c clientset.Interface, namespaces []string, timeout time.Duration) error {
	ginkgo.By(fmt.Sprintf("Waiting for namespaces %+v to vanish", namespaces))
	nsMap := map[string]bool{}
	for _, ns := range namespaces {
		nsMap[ns] = true
	}
	//Now POLL until all namespaces have been eradicated.
	return wait.PollUntilContextTimeout(ctx, 2*time.Second, timeout, false,
		func(ctx context.Context) (bool, error) {
			nsList, err := c.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
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

func waitForConfigMapInNamespace(ctx context.Context, c clientset.Interface, ns, name string, timeout time.Duration) error {
	fieldSelector := fields.OneTermEqualSelector("metadata.name", name).String()
	ctx, cancel := watchtools.ContextWithOptionalTimeout(ctx, timeout)
	defer cancel()
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (object runtime.Object, e error) {
			options.FieldSelector = fieldSelector
			return c.CoreV1().ConfigMaps(ns).List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (i watch.Interface, e error) {
			options.FieldSelector = fieldSelector
			return c.CoreV1().ConfigMaps(ns).Watch(ctx, options)
		},
	}
	_, err := watchtools.UntilWithSync(ctx, lw, &v1.ConfigMap{}, nil, func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Deleted:
			return false, apierrors.NewNotFound(schema.GroupResource{Resource: "configmaps"}, name)
		case watch.Added, watch.Modified:
			return true, nil
		}
		return false, nil
	})
	return err
}

func waitForServiceAccountInNamespace(ctx context.Context, c clientset.Interface, ns, serviceAccountName string, timeout time.Duration) error {
	fieldSelector := fields.OneTermEqualSelector("metadata.name", serviceAccountName).String()
	ctx, cancel := watchtools.ContextWithOptionalTimeout(ctx, timeout)
	defer cancel()
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (object runtime.Object, e error) {
			options.FieldSelector = fieldSelector
			return c.CoreV1().ServiceAccounts(ns).List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (i watch.Interface, e error) {
			options.FieldSelector = fieldSelector
			return c.CoreV1().ServiceAccounts(ns).Watch(ctx, options)
		},
	}
	_, err := watchtools.UntilWithSync(ctx, lw, &v1.ServiceAccount{}, nil, func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Deleted:
			return false, apierrors.NewNotFound(schema.GroupResource{Resource: "serviceaccounts"}, serviceAccountName)
		case watch.Added, watch.Modified:
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		return fmt.Errorf("wait for service account %q in namespace %q: %w", serviceAccountName, ns, err)
	}
	return nil
}

// WaitForDefaultServiceAccountInNamespace waits for the default service account to be provisioned
// the default service account is what is associated with pods when they do not specify a service account
// as a result, pods are not able to be provisioned in a namespace until the service account is provisioned
func WaitForDefaultServiceAccountInNamespace(ctx context.Context, c clientset.Interface, namespace string) error {
	return waitForServiceAccountInNamespace(ctx, c, namespace, defaultServiceAccountName, ServiceAccountProvisionTimeout)
}

// WaitForKubeRootCAInNamespace waits for the configmap kube-root-ca.crt containing the service account
// CA trust bundle to be provisioned in the specified namespace so that pods do not have to retry mounting
// the config map (which creates noise that hides other issues in the Kubelet).
func WaitForKubeRootCAInNamespace(ctx context.Context, c clientset.Interface, namespace string) error {
	return waitForConfigMapInNamespace(ctx, c, namespace, "kube-root-ca.crt", ServiceAccountProvisionTimeout)
}

// CreateTestingNS should be used by every test, note that we append a common prefix to the provided test name.
// Please see NewFramework instead of using this directly.
func CreateTestingNS(ctx context.Context, baseName string, c clientset.Interface, labels map[string]string) (*v1.Namespace, error) {
	if labels == nil {
		labels = map[string]string{}
	}
	labels["e2e-run"] = string(RunID)

	// We don't use ObjectMeta.GenerateName feature, as in case of API call
	// failure we don't know whether the namespace was created and what is its
	// name.
	name := fmt.Sprintf("%v-%v", baseName, RandomSuffix())

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
	if err := wait.PollUntilContextTimeout(ctx, Poll, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		var err error
		got, err = c.CoreV1().Namespaces().Create(ctx, namespaceObj, metav1.CreateOptions{})
		if err != nil {
			if apierrors.IsAlreadyExists(err) {
				// regenerate on conflict
				Logf("Namespace name %q was already taken, generate a new name and retry", namespaceObj.Name)
				namespaceObj.Name = fmt.Sprintf("%v-%v", baseName, RandomSuffix())
			} else {
				Logf("Unexpected error while creating namespace: %v", err)
			}
			return false, nil
		}
		return true, nil
	}); err != nil {
		return nil, err
	}

	if TestContext.VerifyServiceAccount {
		if err := WaitForDefaultServiceAccountInNamespace(ctx, c, got.Name); err != nil {
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
func CheckTestingNSDeletedExcept(ctx context.Context, c clientset.Interface, skip string) error {
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
		namespaces, err := c.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
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

// WaitForServiceEndpointsNum waits until the amount of endpoints that implement service to expectNum.
// Some components use EndpointSlices other Endpoints, we must verify that both objects meet the requirements.
func WaitForServiceEndpointsNum(ctx context.Context, c clientset.Interface, namespace, serviceName string, expectNum int, interval, timeout time.Duration) error {
	return wait.PollUntilContextTimeout(ctx, interval, timeout, false, func(ctx context.Context) (bool, error) {
		Logf("Waiting for amount of service:%s endpoints to be %d", serviceName, expectNum)
		endpoint, err := c.CoreV1().Endpoints(namespace).Get(ctx, serviceName, metav1.GetOptions{})
		if err != nil {
			Logf("Unexpected error trying to get Endpoints for %s : %v", serviceName, err)
			return false, nil
		}

		if countEndpointsNum(endpoint) != expectNum {
			Logf("Unexpected number of Endpoints, got %d, expected %d", countEndpointsNum(endpoint), expectNum)
			return false, nil
		}

		// Endpoints are single family but EndpointSlices can have dual stack addresses,
		// so we verify the number of addresses that matches the same family on both.
		addressType := discoveryv1.AddressTypeIPv4
		if isIPv6Endpoint(endpoint) {
			addressType = discoveryv1.AddressTypeIPv6
		}

		esList, err := c.DiscoveryV1().EndpointSlices(namespace).List(ctx, metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", discoveryv1.LabelServiceName, serviceName)})
		if err != nil {
			Logf("Unexpected error trying to get EndpointSlices for %s : %v", serviceName, err)
			return false, nil
		}

		if len(esList.Items) == 0 {
			Logf("Waiting for at least 1 EndpointSlice to exist")
			return false, nil
		}

		if countEndpointsSlicesNum(esList, addressType) != expectNum {
			Logf("Unexpected number of Endpoints on Slices, got %d, expected %d", countEndpointsSlicesNum(esList, addressType), expectNum)
			return false, nil
		}
		return true, nil
	})
}

func countEndpointsNum(e *v1.Endpoints) int {
	num := 0
	for _, sub := range e.Subsets {
		num += len(sub.Addresses)
	}
	return num
}

// isIPv6Endpoint returns true if the Endpoint uses IPv6 addresses
func isIPv6Endpoint(e *v1.Endpoints) bool {
	for _, sub := range e.Subsets {
		for _, addr := range sub.Addresses {
			if len(addr.IP) == 0 {
				continue
			}
			// Endpoints are single family, so it is enough to check only one address
			return netutils.IsIPv6String(addr.IP)
		}
	}
	// default to IPv4 an Endpoint without IP addresses
	return false
}

func countEndpointsSlicesNum(epList *discoveryv1.EndpointSliceList, addressType discoveryv1.AddressType) int {
	// EndpointSlices can contain the same address on multiple Slices
	addresses := sets.Set[string]{}
	for _, epSlice := range epList.Items {
		if epSlice.AddressType != addressType {
			continue
		}
		for _, ep := range epSlice.Endpoints {
			if len(ep.Addresses) > 0 {
				addresses.Insert(ep.Addresses[0])
			}
		}
	}
	return addresses.Len()
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
			testDesc := ginkgo.CurrentSpecReport()
			if len(testDesc.ContainerHierarchyTexts) > 0 {
				testName := strings.Join(testDesc.ContainerHierarchyTexts, " ")
				if len(testDesc.LeafNodeText) > 0 {
					testName = testName + " " + testDesc.LeafNodeText
				}
				config.UserAgent = fmt.Sprintf("%s -- %s", restclient.DefaultKubernetesUserAgent(), testName)
			}
		}
	}()

	if TestContext.NodeE2E {
		// This is a node e2e test, apply the node e2e configuration
		return &restclient.Config{
			Host:        TestContext.Host,
			BearerToken: TestContext.BearerToken,
			TLSClientConfig: restclient.TLSClientConfig{
				Insecure: true,
			},
		}, nil
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
	// cmd.Args contains command itself as 0th argument, so it's sufficient to
	// print 1st and latter arguments
	Logf("Asynchronously running '%s %s'", cmd.Path, strings.Join(cmd.Args[1:], " "))
	err = cmd.Start()
	return
}

// TryKill is rough equivalent of ctrl+c for cleaning up processes. Intended to be run in defer.
func TryKill(cmd *exec.Cmd) {
	if err := cmd.Process.Kill(); err != nil {
		Logf("ERROR failed to kill command %v! The process may leak", cmd)
	}
}

// EnsureLoadBalancerResourcesDeleted ensures that cloud load balancer resources that were created
// are actually cleaned up.  Currently only implemented for GCE/GKE.
func EnsureLoadBalancerResourcesDeleted(ctx context.Context, ip, portRange string) error {
	return TestContext.CloudConfig.Provider.EnsureLoadBalancerResourcesDeleted(ctx, ip, portRange)
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
	env := os.Environ()
	env = append(env, fmt.Sprintf("LOG_DUMP_SYSTEMD_SERVICES=%s", parseSystemdServices(TestContext.SystemdServices)))
	env = append(env, fmt.Sprintf("LOG_DUMP_SYSTEMD_JOURNAL=%v", TestContext.DumpSystemdJournal))
	cmd.Env = env

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

// GetNodeExternalIPs returns a list of external ip address(es) if any for a node
func GetNodeExternalIPs(node *v1.Node) (ips []string) {
	for j := range node.Status.Addresses {
		nodeAddress := &node.Status.Addresses[j]
		if nodeAddress.Type == v1.NodeExternalIP && nodeAddress.Address != "" {
			ips = append(ips, nodeAddress.Address)
		}
	}
	return
}

// GetControlPlaneNodes returns a list of control plane nodes
func GetControlPlaneNodes(ctx context.Context, c clientset.Interface) *v1.NodeList {
	allNodes, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	ExpectNoError(err, "error reading all nodes")

	var cpNodes v1.NodeList

	for _, node := range allNodes.Items {
		// Check for the control plane label
		if _, hasLabel := node.Labels[ControlPlaneLabel]; hasLabel {
			cpNodes.Items = append(cpNodes.Items, node)
			continue
		}

		// Check for the specific taint
		for _, taint := range node.Spec.Taints {
			// NOTE the taint key is the same as the control plane label
			if taint.Key == ControlPlaneLabel && taint.Effect == v1.TaintEffectNoSchedule {
				cpNodes.Items = append(cpNodes.Items, node)
				continue
			}
		}
	}

	return &cpNodes
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
	return formatted.String()
}

// WatchEventSequenceVerifier ...
// manages a watch for a given resource, ensures that events take place in a given order, retries the test on failure
//
//	ctx                 cancellation signal across API boundaries, e.g: context from Ginkgo
//	dc                  sets up a client to the API
//	resourceType        specify the type of resource
//	namespace           select a namespace
//	resourceName        the name of the given resource
//	listOptions         options used to find the resource, recommended to use listOptions.labelSelector
//	expectedWatchEvents array of events which are expected to occur
//	scenario            the test itself
//	retryCleanup        a function to run which ensures that there are no dangling resources upon test failure
//
// this tooling relies on the test to return the events as they occur
// the entire scenario must be run to ensure that the desired watch events arrive in order (allowing for interweaving of watch events)
//
//	if an expected watch event is missing we elect to clean up and run the entire scenario again
//
// we try the scenario three times to allow the sequencing to fail a couple of times
func WatchEventSequenceVerifier(ctx context.Context, dc dynamic.Interface, resourceType schema.GroupVersionResource, namespace string, resourceName string, listOptions metav1.ListOptions, expectedWatchEvents []watch.Event, scenario func(*watchtools.RetryWatcher) []watch.Event, retryCleanup func() error) {
	listWatcher := &cache.ListWatch{
		WatchFunc: func(listOptions metav1.ListOptions) (watch.Interface, error) {
			return dc.Resource(resourceType).Namespace(namespace).Watch(ctx, listOptions)
		},
	}

	retries := 3
retriesLoop:
	for try := 1; try <= retries; try++ {
		initResource, err := dc.Resource(resourceType).Namespace(namespace).List(ctx, listOptions)
		ExpectNoError(err, "Failed to fetch initial resource")

		resourceWatch, err := watchtools.NewRetryWatcher(initResource.GetResourceVersion(), listWatcher)
		ExpectNoError(err, "Failed to create a resource watch of %v in namespace %v", resourceType.Resource, namespace)

		// NOTE the test may need access to the events to see what's going on, such as a change in status
		actualWatchEvents := scenario(resourceWatch)
		errs := sets.NewString()
		gomega.Expect(len(expectedWatchEvents)).To(gomega.BeNumerically("<=", len(actualWatchEvents)), "Did not get enough watch events")

		totalValidWatchEvents := 0
		foundEventIndexes := map[int]*int{}

		for watchEventIndex, expectedWatchEvent := range expectedWatchEvents {
			foundExpectedWatchEvent := false
		actualWatchEventsLoop:
			for actualWatchEventIndex, actualWatchEvent := range actualWatchEvents {
				if foundEventIndexes[actualWatchEventIndex] != nil {
					continue actualWatchEventsLoop
				}
				if actualWatchEvent.Type == expectedWatchEvent.Type {
					foundExpectedWatchEvent = true
					foundEventIndexes[actualWatchEventIndex] = &watchEventIndex
					break actualWatchEventsLoop
				}
			}
			if !foundExpectedWatchEvent {
				errs.Insert(fmt.Sprintf("Watch event %v not found", expectedWatchEvent.Type))
			}
			totalValidWatchEvents++
		}
		err = retryCleanup()
		ExpectNoError(err, "Error occurred when cleaning up resources")
		if errs.Len() > 0 && try < retries {
			fmt.Println("invariants violated:\n", strings.Join(errs.List(), "\n - "))
			continue retriesLoop
		}
		if errs.Len() > 0 {
			Failf("Unexpected error(s): %v", strings.Join(errs.List(), "\n - "))
		}
		gomega.Expect(expectedWatchEvents).To(gomega.HaveLen(totalValidWatchEvents), "Error: there must be an equal amount of total valid watch events (%d) and expected watch events (%d)", totalValidWatchEvents, len(expectedWatchEvents))
		break retriesLoop
	}
}
