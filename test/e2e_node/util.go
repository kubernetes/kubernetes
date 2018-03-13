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

package e2e_node

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/golang/glog"

	apiv1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes/scheme"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	internalapi "k8s.io/kubernetes/pkg/kubelet/apis/cri"
	stats "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubeletconfigcodec "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/codec"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/remote"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/metrics"
	frameworkmetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// TODO(random-liu): Get this automatically from kubelet flag.
var kubeletAddress = flag.String("kubelet-address", "http://127.0.0.1:10255", "Host and port of the kubelet")

var startServices = flag.Bool("start-services", true, "If true, start local node services")
var stopServices = flag.Bool("stop-services", true, "If true, stop local node services after running tests")
var busyboxImage = imageutils.GetE2EImage(imageutils.BusyBox)

const (
	// Kubelet internal cgroup name for node allocatable cgroup.
	defaultNodeAllocatableCgroup = "kubepods"
)

func getNodeSummary() (*stats.Summary, error) {
	req, err := http.NewRequest("GET", *kubeletAddress+"/stats/summary", nil)
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

// Returns the current KubeletConfiguration
func getCurrentKubeletConfig() (*kubeletconfig.KubeletConfiguration, error) {
	resp := pollConfigz(5*time.Minute, 5*time.Second)
	kubeCfg, err := decodeConfigz(resp)
	if err != nil {
		return nil, err
	}
	return kubeCfg, nil
}

// Must be called within a Context. Allows the function to modify the KubeletConfiguration during the BeforeEach of the context.
// The change is reverted in the AfterEach of the context.
// Returns true on success.
func tempSetCurrentKubeletConfig(f *framework.Framework, updateFunction func(initialConfig *kubeletconfig.KubeletConfiguration)) {
	var oldCfg *kubeletconfig.KubeletConfiguration
	BeforeEach(func() {
		configEnabled, err := isKubeletConfigEnabled(f)
		framework.ExpectNoError(err)
		Expect(configEnabled).To(BeTrue(), "The Dynamic Kubelet Configuration feature is not enabled.\n"+
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
	AfterEach(func() {
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
	src := &apiv1.NodeConfigSource{
		ConfigMap: &apiv1.ConfigMapNodeConfigSource{
			Namespace:        "kube-system",
			Name:             cm.Name,
			KubeletConfigKey: "kubelet",
		},
	}

	// set the source, retry a few times in case we are competing with other writers
	Eventually(func() error {
		if err := setNodeConfigSource(f, src); err != nil {
			return err
		}
		return nil
	}, time.Minute, time.Second).Should(BeNil())

	// poll for new config, for a maximum wait of restartGap
	Eventually(func() error {
		newKubeCfg, err := getCurrentKubeletConfig()
		if err != nil {
			return fmt.Errorf("failed trying to get current Kubelet config, will retry, error: %v", err)
		}
		if !apiequality.Semantic.DeepEqual(*kubeCfg, *newKubeCfg) {
			return fmt.Errorf("still waiting for new configuration to take effect, will continue to watch /configz")
		}
		glog.Infof("new configuration has taken effect")
		return nil
	}, restartGap, pollInterval).Should(BeNil())

	return nil
}

// sets the current node's configSource, this should only be called from Serial tests
func setNodeConfigSource(f *framework.Framework, source *apiv1.NodeConfigSource) error {
	// since this is a serial test, we just get the node, change the source, and then update it
	// this prevents any issues with the patch API from affecting the test results
	nodeclient := f.ClientSet.CoreV1().Nodes()

	// get the node
	node, err := nodeclient.Get(framework.TestContext.NodeName, metav1.GetOptions{})
	if err != nil {
		return err
	}

	// set new source
	node.Spec.ConfigSource = source

	// update to the new source
	_, err = nodeclient.Update(node)
	if err != nil {
		return err
	}

	return nil
}

// Causes the test to fail, or returns a status 200 response from the /configz endpoint
func pollConfigz(timeout time.Duration, pollInterval time.Duration) *http.Response {
	endpoint := fmt.Sprintf("http://127.0.0.1:8080/api/v1/nodes/%s/proxy/configz", framework.TestContext.NodeName)
	client := &http.Client{}
	req, err := http.NewRequest("GET", endpoint, nil)
	framework.ExpectNoError(err)
	req.Header.Add("Accept", "application/json")

	var resp *http.Response
	Eventually(func() bool {
		resp, err = client.Do(req)
		if err != nil {
			glog.Errorf("Failed to get /configz, retrying. Error: %v", err)
			return false
		}
		if resp.StatusCode != 200 {
			glog.Errorf("/configz response status not 200, retrying. Response was: %+v", resp)
			return false
		}
		return true
	}, timeout, pollInterval).Should(Equal(true))
	return resp
}

// Decodes the http response from /configz and returns a kubeletconfig.KubeletConfiguration (internal type).
func decodeConfigz(resp *http.Response) (*kubeletconfig.KubeletConfiguration, error) {
	// This hack because /configz reports the following structure:
	// {"kubeletconfig": {the JSON representation of kubeletconfigv1beta1.KubeletConfiguration}}
	type configzWrapper struct {
		ComponentConfig kubeletconfigv1beta1.KubeletConfiguration `json:"kubeletconfig"`
	}

	configz := configzWrapper{}
	kubeCfg := kubeletconfig.KubeletConfiguration{}

	contentsBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal(contentsBytes, &configz)
	if err != nil {
		return nil, err
	}

	err = scheme.Scheme.Convert(&configz.ComponentConfig, &kubeCfg, nil)
	if err != nil {
		return nil, err
	}

	return &kubeCfg, nil
}

// creates a configmap containing kubeCfg in kube-system namespace
func createConfigMap(f *framework.Framework, internalKC *kubeletconfig.KubeletConfiguration) (*apiv1.ConfigMap, error) {
	cmap := newKubeletConfigMap("testcfg", internalKC)
	cmap, err := f.ClientSet.CoreV1().ConfigMaps("kube-system").Create(cmap)
	if err != nil {
		return nil, err
	}
	return cmap, nil
}

// constructs a ConfigMap, populating one of its keys with the KubeletConfiguration. Always uses GenerateName to generate a suffix.
func newKubeletConfigMap(name string, internalKC *kubeletconfig.KubeletConfiguration) *apiv1.ConfigMap {
	data, err := kubeletconfigcodec.EncodeKubeletConfig(internalKC, kubeletconfigv1beta1.SchemeGroupVersion)
	framework.ExpectNoError(err)

	cmap := &apiv1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{GenerateName: name + "-"},
		Data: map[string]string{
			"kubelet": string(data),
		},
	}
	return cmap
}

func logPodEvents(f *framework.Framework) {
	framework.Logf("Summary of pod events during the test:")
	err := framework.ListNamespaceEvents(f.ClientSet, f.Namespace.Name)
	framework.ExpectNoError(err)
}

func logNodeEvents(f *framework.Framework) {
	framework.Logf("Summary of node events during the test:")
	err := framework.ListNamespaceEvents(f.ClientSet, "")
	framework.ExpectNoError(err)
}

func getLocalNode(f *framework.Framework) *apiv1.Node {
	nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
	Expect(len(nodeList.Items)).To(Equal(1), "Unexpected number of node objects for node e2e. Expects only one node.")
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
	metric, err := metrics.GrabKubeletMetricsWithoutProxy(framework.TestContext.NodeName + ":10255")
	if err != nil {
		framework.Logf("Error getting kubelet metrics: %v", err)
	} else {
		framework.Logf("Kubelet Metrics: %+v", framework.GetKubeletLatencyMetrics(metric, metricSet))
	}
}

// returns config related metrics from the local kubelet, filtered to the filterMetricNames passed in
func getKubeletMetrics(filterMetricNames sets.String) (frameworkmetrics.KubeletMetrics, error) {
	// grab Kubelet metrics
	ms, err := metrics.GrabKubeletMetricsWithoutProxy(framework.TestContext.NodeName + ":10255")
	if err != nil {
		return nil, err
	}

	filtered := metrics.NewKubeletMetrics()
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
func restartKubelet() {
	stdout, err := exec.Command("sudo", "systemctl", "list-units", "kubelet*", "--state=running").CombinedOutput()
	framework.ExpectNoError(err)
	regex := regexp.MustCompile("(kubelet-\\w+)")
	matches := regex.FindStringSubmatch(string(stdout))
	Expect(len(matches)).NotTo(BeZero())
	kube := matches[0]
	framework.Logf("Get running kubelet with systemctl: %v, %v", string(stdout), kube)
	stdout, err = exec.Command("sudo", "systemctl", "restart", kube).CombinedOutput()
	framework.ExpectNoError(err, "Failed to restart kubelet with systemctl: %v, %v", err, stdout)
}

func toCgroupFsName(cgroupName cm.CgroupName) string {
	if framework.TestContext.KubeletConfig.CgroupDriver == "systemd" {
		return cgroupName.ToSystemd()
	} else {
		return cgroupName.ToCgroupfs()
	}
}

// reduceAllocatableMemoryUsage uses memory.force_empty (https://lwn.net/Articles/432224/)
// to make the kernel reclaim memory in the allocatable cgroup
// the time to reduce pressure may be unbounded, but usually finishes within a second
func reduceAllocatableMemoryUsage() {
	cmd := fmt.Sprintf("echo 0 > /sys/fs/cgroup/memory/%s/memory.force_empty", toCgroupFsName(cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup)))
	_, err := exec.Command("sudo", "sh", "-c", cmd).CombinedOutput()
	framework.ExpectNoError(err)
}
