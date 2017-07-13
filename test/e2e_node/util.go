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
	"reflect"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubeletconfigv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1alpha1"
	stats "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/metrics"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// TODO(random-liu): Get this automatically from kubelet flag.
var kubeletAddress = flag.String("kubelet-address", "http://127.0.0.1:10255", "Host and port of the kubelet")

var startServices = flag.Bool("start-services", true, "If true, start local node services")
var stopServices = flag.Bool("stop-services", true, "If true, stop local node services after running tests")

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
		if configEnabled {
			oldCfg, err = getCurrentKubeletConfig()
			framework.ExpectNoError(err)
			clone, err := scheme.Scheme.DeepCopy(oldCfg)
			framework.ExpectNoError(err)
			newCfg := clone.(*kubeletconfig.KubeletConfiguration)
			updateFunction(newCfg)
			framework.ExpectNoError(setKubeletConfiguration(f, newCfg))
		} else {
			framework.Logf("The Dynamic Kubelet Configuration feature is not enabled.\n" +
				"Pass --feature-gates=DynamicKubeletConfig=true to the Kubelet to enable this feature.\n" +
				"For `make test-e2e-node`, you can set `TEST_ARGS='--feature-gates=DynamicKubeletConfig=true'`.")
		}
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
	return strings.Contains(cfgz.FeatureGates, "DynamicKubeletConfig=true"), nil
}

// Creates or updates the configmap for KubeletConfiguration, waits for the Kubelet to restart
// with the new configuration. Returns an error if the configuration after waiting for restartGap
// doesn't match what you attempted to set, or if the dynamic configuration feature is disabled.
func setKubeletConfiguration(f *framework.Framework, kubeCfg *kubeletconfig.KubeletConfiguration) error {
	const (
		restartGap   = 40 * time.Second
		pollInterval = 5 * time.Second
	)

	// make sure Dynamic Kubelet Configuration feature is enabled on the Kubelet we are about to reconfigure
	configEnabled, err := isKubeletConfigEnabled(f)
	if err != nil {
		return fmt.Errorf("could not determine whether 'DynamicKubeletConfig' feature is enabled, err: %v", err)
	}
	if !configEnabled {
		return fmt.Errorf("The Dynamic Kubelet Configuration feature is not enabled.\n" +
			"Pass --feature-gates=DynamicKubeletConfig=true to the Kubelet to enable this feature.\n" +
			"For `make test-e2e-node`, you can set `TEST_ARGS='--feature-gates=DynamicKubeletConfig=true'`.")
	}

	nodeclient := f.ClientSet.CoreV1().Nodes()

	// create the ConfigMap with the new configuration
	cm, err := createConfigMap(f, kubeCfg)
	if err != nil {
		return err
	}

	// create the correct reference object
	src := &v1.NodeConfigSource{
		ConfigMapRef: &v1.ObjectReference{
			Namespace: "kube-system",
			Name:      cm.Name,
			UID:       cm.UID,
		},
	}

	// serialize the new node config source
	raw, err := json.Marshal(src)
	framework.ExpectNoError(err)
	data := []byte(fmt.Sprintf(`{"spec":{"configSource":%s}}`, raw))

	// patch the node
	_, err = nodeclient.Patch(framework.TestContext.NodeName,
		types.StrategicMergePatchType,
		data)
	framework.ExpectNoError(err)

	// poll for new config, for a maximum wait of restartGap
	Eventually(func() error {
		newKubeCfg, err := getCurrentKubeletConfig()
		if err != nil {
			return fmt.Errorf("failed trying to get current Kubelet config, will retry, error: %v", err)
		}
		if !reflect.DeepEqual(*kubeCfg, *newKubeCfg) {
			return fmt.Errorf("still waiting for new configuration to take effect, will continue to watch /configz")
		}
		glog.Infof("new configuration has taken effect")
		return nil
	}, restartGap, pollInterval).Should(BeNil())

	return nil
}

// Causes the test to fail, or returns a status 200 response from the /configz endpoint
func pollConfigz(timeout time.Duration, pollInterval time.Duration) *http.Response {
	endpoint := fmt.Sprintf("http://127.0.0.1:8080/api/v1/proxy/nodes/%s/configz", framework.TestContext.NodeName)
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
	// {"kubeletconfig": {the JSON representation of kubeletconfigv1alpha1.KubeletConfiguration}}
	type configzWrapper struct {
		ComponentConfig kubeletconfigv1alpha1.KubeletConfiguration `json:"kubeletconfig"`
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
func createConfigMap(f *framework.Framework, internalKC *kubeletconfig.KubeletConfiguration) (*v1.ConfigMap, error) {
	cmap := makeKubeletConfigMap(internalKC)
	cmap, err := f.ClientSet.Core().ConfigMaps("kube-system").Create(cmap)
	if err != nil {
		return nil, err
	}
	return cmap, nil
}

// constructs a ConfigMap, populating one of its keys with the KubeletConfiguration. Uses GenerateName.
func makeKubeletConfigMap(internalKC *kubeletconfig.KubeletConfiguration) *v1.ConfigMap {
	externalKC := &kubeletconfigv1alpha1.KubeletConfiguration{}
	api.Scheme.Convert(internalKC, externalKC, nil)

	encoder, err := newJSONEncoder(kubeletconfig.GroupName)
	framework.ExpectNoError(err)

	data, err := runtime.Encode(encoder, externalKC)
	framework.ExpectNoError(err)

	cmap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{GenerateName: "testcfg"},
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

func getLocalNode(f *framework.Framework) *v1.Node {
	nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
	Expect(len(nodeList.Items)).To(Equal(1), "Unexpected number of node objects for node e2e. Expects only one node.")
	return &nodeList.Items[0]
}

// logs prometheus metrics from the local kubelet.
func logKubeletMetrics(metricKeys ...string) {
	metricSet := sets.NewString()
	for _, key := range metricKeys {
		metricSet.Insert(kubeletmetrics.KubeletSubsystem + "_" + key)
	}
	metric, err := metrics.GrabKubeletMetricsWithoutProxy(framework.TestContext.NodeName + ":10255")
	if err != nil {
		framework.Logf("Error getting kubelet metrics: %v", err)
	} else {
		framework.Logf("Kubelet Metrics: %+v", framework.GetKubeletMetrics(metric, metricSet))
	}
}

func newJSONEncoder(groupName string) (runtime.Encoder, error) {
	// encode to json
	mediaType := "application/json"
	info, ok := runtime.SerializerInfoForMediaType(api.Codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return nil, fmt.Errorf("unsupported media type %q", mediaType)
	}

	versions := api.Registry.EnabledVersionsForGroup(groupName)
	if len(versions) == 0 {
		return nil, fmt.Errorf("no enabled versions for group %q", groupName)
	}

	// the "best" version supposedly comes first in the list returned from api.Registry.EnabledVersionsForGroup
	return api.Codecs.EncoderForVersion(info.Serializer, versions[0]), nil
}
