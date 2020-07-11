/*
Copyright 2016 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os/exec"
	"regexp"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/featuregate"
	internalapi "k8s.io/cri-api/pkg/apis"
	"k8s.io/klog/v2"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeletpodresourcesv1alpha1 "k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
	stats "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/cri/remote"
	kubeletconfigcodec "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/codec"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var startServices = flag.Bool("start-services", true, "If true, start local node services")
var stopServices = flag.Bool("stop-services", true, "If true, stop local node services after running tests")
var busyboxImage = imageutils.GetE2EImage(imageutils.BusyBox)

const (
	// Kubelet internal cgroup name for node allocatable cgroup.
	defaultNodeAllocatableCgroup = "kubepods"
	// defaultPodResourcesPath is the path to the local endpoint serving the podresources GRPC service.
	defaultPodResourcesPath    = "/var/lib/kubelet/pod-resources"
	defaultPodResourcesTimeout = 10 * time.Second
	defaultPodResourcesMaxSize = 1024 * 1024 * 16 // 16 Mb
)

func getNodeSummary() (*stats.Summary, error) {
	kubeletConfig, err := getCurrentKubeletConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get current kubelet config")
	}
	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s:%d/stats/summary", kubeletConfig.Address, kubeletConfig.ReadOnlyPort), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to build http request: %v", err)
	}
	req.Header.Add("Accept", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get /stats/summary: %v", err)
	}

	defer resp.Body.Close()
	contentsBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read /stats/summary: %+v", resp)
	}

	decoder := json.NewDecoder(strings.NewReader(string(contentsBytes)))
	summary := stats.Summary{}
	err = decoder.Decode(&summary)
	if err != nil {
		return nil, fmt.Errorf("failed to parse /stats/summary to go struct: %+v", resp)
	}
	return &summary, nil
}

func getNodeDevices() (*kubeletpodresourcesv1alpha1.ListPodResourcesResponse, error) {
	endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
	if err != nil {
		return nil, fmt.Errorf("Error getting local endpoint: %v", err)
	}
	client, conn, err := podresources.GetClient(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
	if err != nil {
		return nil, fmt.Errorf("Error getting grpc client: %v", err)
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	resp, err := client.List(ctx, &kubeletpodresourcesv1alpha1.ListPodResourcesRequest{})
	if err != nil {
		return nil, fmt.Errorf("%v.Get(_) = _, %v", client, err)
	}
	return resp, nil
}

// Returns the current KubeletConfiguration
func getCurrentKubeletConfig() (*kubeletconfig.KubeletConfiguration, error) {
	// namespace only relevant if useProxy==true, so we don't bother
	return e2ekubelet.GetCurrentKubeletConfig(framework.TestContext.NodeName, "", false)
}

// Must be called within a Context. Allows the function to modify the KubeletConfiguration during the BeforeEach of the context.
// The change is reverted in the AfterEach of the context.
// Returns true on success.
func tempSetCurrentKubeletConfig(f *framework.Framework, updateFunction func(initialConfig *kubeletconfig.KubeletConfiguration)) {
	var oldCfg *kubeletconfig.KubeletConfiguration
	ginkgo.BeforeEach(func() {
		configEnabled, err := isKubeletConfigEnabled(f)
		framework.ExpectNoError(err)
		framework.ExpectEqual(configEnabled, true, "The Dynamic Kubelet Configuration feature is not enabled.\n"+
			"Pass --feature-gates=DynamicKubeletConfig=true to the Kubelet to enable this feature.\n"+
			"For `make test-e2e-node`, you can set `TEST_ARGS='--feature-gates=DynamicKubeletConfig=true'`.")
		oldCfg, err = getCurrentKubeletConfig()
		framework.ExpectNoError(err)
		newCfg := oldCfg.DeepCopy()
		updateFunction(newCfg)
		if apiequality.Semantic.DeepEqual(*newCfg, *oldCfg) {
			return
		}

		framework.ExpectNoError(setKubeletConfiguration(f, newCfg))
	})
	ginkgo.AfterEach(func() {
		if oldCfg != nil {
			err := setKubeletConfiguration(f, oldCfg)
			framework.ExpectNoError(err)
		}
	})
}

// Returns true if kubeletConfig is enabled, false otherwise or if we cannot determine if it is.
func isKubeletConfigEnabled(f *framework.Framework) (bool, error) {
	cfgz, err := getCurrentKubeletConfig()
	if err != nil {
		return false, fmt.Errorf("could not determine whether 'DynamicKubeletConfig' feature is enabled, err: %v", err)
	}
	v, ok := cfgz.FeatureGates[string(features.DynamicKubeletConfig)]
	if !ok {
		return true, nil
	}
	return v, nil
}

// Creates or updates the configmap for KubeletConfiguration, waits for the Kubelet to restart
// with the new configuration. Returns an error if the configuration after waiting for restartGap
// doesn't match what you attempted to set, or if the dynamic configuration feature is disabled.
// You should only call this from serial tests.
func setKubeletConfiguration(f *framework.Framework, kubeCfg *kubeletconfig.KubeletConfiguration) error {
	const (
		restartGap   = 40 * time.Second
		pollInterval = 5 * time.Second
	)

	// make sure Dynamic Kubelet Configuration feature is enabled on the Kubelet we are about to reconfigure
	if configEnabled, err := isKubeletConfigEnabled(f); err != nil {
		return err
	} else if !configEnabled {
		return fmt.Errorf("The Dynamic Kubelet Configuration feature is not enabled.\n" +
			"Pass --feature-gates=DynamicKubeletConfig=true to the Kubelet to enable this feature.\n" +
			"For `make test-e2e-node`, you can set `TEST_ARGS='--feature-gates=DynamicKubeletConfig=true'`.")
	}

	// create the ConfigMap with the new configuration
	cm, err := createConfigMap(f, kubeCfg)
	if err != nil {
		return err
	}

	// create the reference and set Node.Spec.ConfigSource
	src := &v1.NodeConfigSource{
		ConfigMap: &v1.ConfigMapNodeConfigSource{
			Namespace:        "kube-system",
			Name:             cm.Name,
			KubeletConfigKey: "kubelet",
		},
	}

	// set the source, retry a few times in case we are competing with other writers
	gomega.Eventually(func() error {
		if err := setNodeConfigSource(f, src); err != nil {
			return err
		}
		return nil
	}, time.Minute, time.Second).Should(gomega.BeNil())

	// poll for new config, for a maximum wait of restartGap
	gomega.Eventually(func() error {
		newKubeCfg, err := getCurrentKubeletConfig()
		if err != nil {
			return fmt.Errorf("failed trying to get current Kubelet config, will retry, error: %v", err)
		}
		if !apiequality.Semantic.DeepEqual(*kubeCfg, *newKubeCfg) {
			return fmt.Errorf("still waiting for new configuration to take effect, will continue to watch /configz")
		}
		klog.Infof("new configuration has taken effect")
		return nil
	}, restartGap, pollInterval).Should(gomega.BeNil())

	return nil
}

// sets the current node's configSource, this should only be called from Serial tests
func setNodeConfigSource(f *framework.Framework, source *v1.NodeConfigSource) error {
	// since this is a serial test, we just get the node, change the source, and then update it
	// this prevents any issues with the patch API from affecting the test results
	nodeclient := f.ClientSet.CoreV1().Nodes()

	// get the node
	node, err := nodeclient.Get(context.TODO(), framework.TestContext.NodeName, metav1.GetOptions{})
	if err != nil {
		return err
	}

	// set new source
	node.Spec.ConfigSource = source

	// update to the new source
	_, err = nodeclient.Update(context.TODO(), node, metav1.UpdateOptions{})
	if err != nil {
		return err
	}

	return nil
}

// creates a configmap containing kubeCfg in kube-system namespace
func createConfigMap(f *framework.Framework, internalKC *kubeletconfig.KubeletConfiguration) (*v1.ConfigMap, error) {
	cmap := newKubeletConfigMap("testcfg", internalKC)
	cmap, err := f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(context.TODO(), cmap, metav1.CreateOptions{})
	if err != nil {
		return nil, err
	}
	return cmap, nil
}

// constructs a ConfigMap, populating one of its keys with the KubeletConfiguration. Always uses GenerateName to generate a suffix.
func newKubeletConfigMap(name string, internalKC *kubeletconfig.KubeletConfiguration) *v1.ConfigMap {
	data, err := kubeletconfigcodec.EncodeKubeletConfig(internalKC, kubeletconfigv1beta1.SchemeGroupVersion)
	framework.ExpectNoError(err)

	cmap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{GenerateName: name + "-"},
		Data: map[string]string{
			"kubelet": string(data),
		},
	}
	return cmap
}

// listNamespaceEvents lists the events in the given namespace.
func listNamespaceEvents(c clientset.Interface, ns string) error {
	ls, err := c.CoreV1().Events(ns).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return err
	}
	for _, event := range ls.Items {
		klog.Infof("Event(%#v): type: '%v' reason: '%v' %v", event.InvolvedObject, event.Type, event.Reason, event.Message)
	}
	return nil
}

func logPodEvents(f *framework.Framework) {
	framework.Logf("Summary of pod events during the test:")
	err := listNamespaceEvents(f.ClientSet, f.Namespace.Name)
	framework.ExpectNoError(err)
}

func logNodeEvents(f *framework.Framework) {
	framework.Logf("Summary of node events during the test:")
	err := listNamespaceEvents(f.ClientSet, "")
	framework.ExpectNoError(err)
}

func getLocalNode(f *framework.Framework) *v1.Node {
	nodeList, err := e2enode.GetReadySchedulableNodes(f.ClientSet)
	framework.ExpectNoError(err)
	framework.ExpectEqual(len(nodeList.Items), 1, "Unexpected number of node objects for node e2e. Expects only one node.")
	return &nodeList.Items[0]
}

// logKubeletLatencyMetrics logs KubeletLatencyMetrics computed from the Prometheus
// metrics exposed on the current node and identified by the metricNames.
// The Kubelet subsystem prefix is automatically prepended to these metric names.
func logKubeletLatencyMetrics(metricNames ...string) {
	metricSet := sets.NewString()
	for _, key := range metricNames {
		metricSet.Insert(kubeletmetrics.KubeletSubsystem + "_" + key)
	}
	metric, err := e2emetrics.GrabKubeletMetricsWithoutProxy(framework.TestContext.NodeName+":10255", "/metrics")
	if err != nil {
		framework.Logf("Error getting kubelet metrics: %v", err)
	} else {
		framework.Logf("Kubelet Metrics: %+v", e2emetrics.GetKubeletLatencyMetrics(metric, metricSet))
	}
}

// returns config related metrics from the local kubelet, filtered to the filterMetricNames passed in
func getKubeletMetrics(filterMetricNames sets.String) (e2emetrics.KubeletMetrics, error) {
	// grab Kubelet metrics
	ms, err := e2emetrics.GrabKubeletMetricsWithoutProxy(framework.TestContext.NodeName+":10255", "/metrics")
	if err != nil {
		return nil, err
	}

	filtered := e2emetrics.NewKubeletMetrics()
	for name := range ms {
		if !filterMetricNames.Has(name) {
			continue
		}
		filtered[name] = ms[name]
	}
	return filtered, nil
}

// runCommand runs the cmd and returns the combined stdout and stderr, or an
// error if the command failed.
func runCommand(cmd ...string) (string, error) {
	output, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to run %q: %s (%s)", strings.Join(cmd, " "), err, output)
	}
	return string(output), nil
}

// getCRIClient connects CRI and returns CRI runtime service clients and image service client.
func getCRIClient() (internalapi.RuntimeService, internalapi.ImageManagerService, error) {
	// connection timeout for CRI service connection
	const connectionTimeout = 2 * time.Minute
	runtimeEndpoint := framework.TestContext.ContainerRuntimeEndpoint
	r, err := remote.NewRemoteRuntimeService(runtimeEndpoint, connectionTimeout)
	if err != nil {
		return nil, nil, err
	}
	imageManagerEndpoint := runtimeEndpoint
	if framework.TestContext.ImageServiceEndpoint != "" {
		//ImageServiceEndpoint is the same as ContainerRuntimeEndpoint if not
		//explicitly specified
		imageManagerEndpoint = framework.TestContext.ImageServiceEndpoint
	}
	i, err := remote.NewRemoteImageService(imageManagerEndpoint, connectionTimeout)
	if err != nil {
		return nil, nil, err
	}
	return r, i, nil
}

// TODO: Find a uniform way to deal with systemctl/initctl/service operations. #34494
func findRunningKubletServiceName() string {
	stdout, err := exec.Command("sudo", "systemctl", "list-units", "*kubelet*", "--state=running").CombinedOutput()
	framework.ExpectNoError(err)
	regex := regexp.MustCompile("(kubelet-\\w+)")
	matches := regex.FindStringSubmatch(string(stdout))
	framework.ExpectNotEqual(len(matches), 0, "Found more than one kubelet service running: %q", stdout)
	kubeletServiceName := matches[0]
	framework.Logf("Get running kubelet with systemctl: %v, %v", string(stdout), kubeletServiceName)
	return kubeletServiceName
}

func restartKubelet() {
	kubeletServiceName := findRunningKubletServiceName()
	// reset the kubelet service start-limit-hit
	stdout, err := exec.Command("sudo", "systemctl", "reset-failed", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to reset kubelet start-limit-hit with systemctl: %v, %v", err, stdout)

	stdout, err = exec.Command("sudo", "systemctl", "restart", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to restart kubelet with systemctl: %v, %v", err, stdout)
}

// stopKubelet will kill the running kubelet, and returns a func that will restart the process again
func stopKubelet() func() {
	kubeletServiceName := findRunningKubletServiceName()
	stdout, err := exec.Command("sudo", "systemctl", "kill", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to stop kubelet with systemctl: %v, %v", err, stdout)
	return func() {
		stdout, err := exec.Command("sudo", "systemctl", "start", kubeletServiceName).CombinedOutput()
		framework.ExpectNoError(err, "Failed to restart kubelet with systemctl: %v, %v", err, stdout)
	}
}

func toCgroupFsName(cgroupName cm.CgroupName) string {
	if framework.TestContext.KubeletConfig.CgroupDriver == "systemd" {
		return cgroupName.ToSystemd()
	}
	return cgroupName.ToCgroupfs()
}

// reduceAllocatableMemoryUsage uses memory.force_empty (https://lwn.net/Articles/432224/)
// to make the kernel reclaim memory in the allocatable cgroup
// the time to reduce pressure may be unbounded, but usually finishes within a second
func reduceAllocatableMemoryUsage() {
	cmd := fmt.Sprintf("echo 0 > /sys/fs/cgroup/memory/%s/memory.force_empty", toCgroupFsName(cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup)))
	_, err := exec.Command("sudo", "sh", "-c", cmd).CombinedOutput()
	framework.ExpectNoError(err)
}

// Equivalent of featuregatetesting.SetFeatureGateDuringTest
// which can't be used here because we're not in a Testing context.
// This must be in a non-"_test" file to pass
// make verify WHAT=test-featuregates
func withFeatureGate(feature featuregate.Feature, desired bool) func() {
	current := utilfeature.DefaultFeatureGate.Enabled(feature)
	utilfeature.DefaultMutableFeatureGate.Set(fmt.Sprintf("%s=%v", string(feature), desired))
	return func() {
		utilfeature.DefaultMutableFeatureGate.Set(fmt.Sprintf("%s=%v", string(feature), current))
	}
}
