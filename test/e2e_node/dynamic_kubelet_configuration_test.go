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
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// This test is marked [Disruptive] because the Kubelet temporarily goes down as part of of this test.
var _ = framework.KubeDescribe("DynamicKubeletConfiguration [Feature:DynamicKubeletConfig] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("dynamic-kubelet-configuration-test")

	Context("When a configmap called `kubelet-<node-name>` is added to the `kube-system` namespace", func() {
		It("The Kubelet on that node should restart to take up the new config", func() {
			const (
				restartGap = 40 * time.Second
			)

			// Get the current KubeletConfiguration (known to be valid) by
			// querying the configz endpoint for the current node.
			resp := pollConfigz(2*time.Minute, 5*time.Second)
			kubeCfg, err := decodeConfigz(resp)
			framework.ExpectNoError(err)
			glog.Infof("KubeletConfiguration - Initial values: %+v", *kubeCfg)

			// Change a safe value e.g. file check frequency.
			// Make sure we're providing a value distinct from the current one.
			oldFileCheckFrequency := kubeCfg.FileCheckFrequency.Duration
			newFileCheckFrequency := 11 * time.Second
			if kubeCfg.FileCheckFrequency.Duration == newFileCheckFrequency {
				newFileCheckFrequency = 10 * time.Second
			}
			kubeCfg.FileCheckFrequency.Duration = newFileCheckFrequency

			// Use the new config to create a new kube-<node-name> configmap in `kube-system` namespace.
			_, err = createConfigMap(f, kubeCfg)
			framework.ExpectNoError(err)

			// Give the Kubelet time to see that there is new config and restart. If we don't do this,
			// the Kubelet will still have the old config when we poll, and the test will fail.
			time.Sleep(restartGap)

			// Use configz to get the new config.
			resp = pollConfigz(2*time.Minute, 5*time.Second)
			kubeCfg, err = decodeConfigz(resp)
			framework.ExpectNoError(err)
			glog.Infof("KubeletConfiguration - After modification of FileCheckFrequency: %+v", *kubeCfg)

			// We expect to see the new value in the new config.
			Expect(kubeCfg.FileCheckFrequency.Duration).To(Equal(newFileCheckFrequency))

			// Change the config back to what it originally was.
			kubeCfg.FileCheckFrequency.Duration = oldFileCheckFrequency
			_, err = updateConfigMap(f, kubeCfg)
			framework.ExpectNoError(err)

			// Give the Kubelet time to see that there is new config and restart. If we don't do this,
			// the Kubelet will still have the old config when we poll, and the test will fail.
			time.Sleep(restartGap)

			// User configz to get the new config.
			resp = pollConfigz(2*time.Minute, 5*time.Second)
			kubeCfg, err = decodeConfigz(resp)
			framework.ExpectNoError(err)
			glog.Infof("KubeletConfiguration - After restoration of FileCheckFrequency: %+v", *kubeCfg)

			// We expect to see the original value restored in the new config.
			Expect(kubeCfg.FileCheckFrequency.Duration).To(Equal(oldFileCheckFrequency))
		})
	})
})

// This function either causes the test to fail, or it returns a status 200 response.
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

// Decodes the http response  from /configz and returns a componentconfig.KubeletConfiguration (internal type).
func decodeConfigz(resp *http.Response) (*componentconfig.KubeletConfiguration, error) {
	// This hack because /configz reports the following structure:
	// {"componentconfig": {the JSON representation of v1alpha1.KubeletConfiguration}}
	type configzWrapper struct {
		ComponentConfig v1alpha1.KubeletConfiguration `json:"componentconfig"`
	}

	configz := configzWrapper{}
	kubeCfg := componentconfig.KubeletConfiguration{}

	contentsBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal(contentsBytes, &configz)
	if err != nil {
		return nil, err
	}

	err = api.Scheme.Convert(&configz.ComponentConfig, &kubeCfg, nil)
	if err != nil {
		return nil, err
	}

	return &kubeCfg, nil
}

// Uses KubeletConfiguration to create a `kubelet-<node-name>` ConfigMap in the "kube-system" namespace.
func createConfigMap(f *framework.Framework, kubeCfg *componentconfig.KubeletConfiguration) (*api.ConfigMap, error) {
	kubeCfgExt := v1alpha1.KubeletConfiguration{}
	api.Scheme.Convert(kubeCfg, &kubeCfgExt, nil)

	bytes, err := json.Marshal(kubeCfgExt)
	framework.ExpectNoError(err)

	cmap, err := f.Client.ConfigMaps("kube-system").Create(&api.ConfigMap{
		ObjectMeta: api.ObjectMeta{
			Name: fmt.Sprintf("kubelet-%s", framework.TestContext.NodeName),
		},
		Data: map[string]string{
			"kubelet.config": string(bytes),
		},
	})
	if err != nil {
		return nil, err
	}
	return cmap, nil
}

// Similar to createConfigMap, except this updates an existing ConfigMap.
func updateConfigMap(f *framework.Framework, kubeCfg *componentconfig.KubeletConfiguration) (*api.ConfigMap, error) {
	kubeCfgExt := v1alpha1.KubeletConfiguration{}
	api.Scheme.Convert(kubeCfg, &kubeCfgExt, nil)

	bytes, err := json.Marshal(kubeCfgExt)
	framework.ExpectNoError(err)

	cmap, err := f.Client.ConfigMaps("kube-system").Update(&api.ConfigMap{
		ObjectMeta: api.ObjectMeta{
			Name: fmt.Sprintf("kubelet-%s", framework.TestContext.NodeName),
		},
		Data: map[string]string{
			"kubelet.config": string(bytes),
		},
	})
	if err != nil {
		return nil, err
	}
	return cmap, nil
}
