/*
Copyright 2021 The Kubernetes Authors.

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

package kubeletconfig

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"time"

	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/util/wait"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	"k8s.io/kubernetes/pkg/cluster/ports"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletconfigscheme "k8s.io/kubernetes/pkg/kubelet/apis/config/scheme"
	kubeletconfigcodec "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/codec"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"

	"sigs.k8s.io/yaml"
)

func getKubeletConfigFilePath() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get current working directory: %w", err)
	}

	// DO NOT name this file "kubelet" - you will overwrite the kubelet binary and be very confused :)
	return filepath.Join(cwd, "kubelet-config"), nil
}

// GetCurrentKubeletConfigFromFile returns the current kubelet configuration under the filesystem.
// This method should only run together with e2e node tests, meaning the test executor and the cluster nodes is the
// same machine
func GetCurrentKubeletConfigFromFile() (*kubeletconfig.KubeletConfiguration, error) {
	kubeletConfigFilePath, err := getKubeletConfigFilePath()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(kubeletConfigFilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to get the kubelet config from the file %q: %w", kubeletConfigFilePath, err)
	}

	var kubeletConfigV1Beta1 kubeletconfigv1beta1.KubeletConfiguration
	if err := yaml.Unmarshal(data, &kubeletConfigV1Beta1); err != nil {
		return nil, fmt.Errorf("failed to unmarshal the kubelet config: %w", err)
	}

	scheme, _, err := kubeletconfigscheme.NewSchemeAndCodecs()
	if err != nil {
		return nil, err
	}

	kubeletConfig := kubeletconfig.KubeletConfiguration{}
	err = scheme.Convert(&kubeletConfigV1Beta1, &kubeletConfig, nil)
	if err != nil {
		return nil, err
	}

	return &kubeletConfig, nil
}

// WriteKubeletConfigFile updates the kubelet configuration under the filesystem
// This method should only run together with e2e node tests, meaning the test executor and the cluster nodes is the
// same machine
func WriteKubeletConfigFile(kubeletConfig *kubeletconfig.KubeletConfiguration) error {
	data, err := kubeletconfigcodec.EncodeKubeletConfig(kubeletConfig, kubeletconfigv1beta1.SchemeGroupVersion)
	if err != nil {
		return err
	}

	kubeletConfigFilePath, err := getKubeletConfigFilePath()
	if err != nil {
		return err
	}

	if err := os.WriteFile(kubeletConfigFilePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write the kubelet file to %q: %w", kubeletConfigFilePath, err)
	}

	return nil
}

// GetCurrentKubeletConfig fetches the current Kubelet Config for the given node
func GetCurrentKubeletConfig(ctx context.Context, nodeName, namespace string, useProxy bool, standaloneMode bool) (*kubeletconfig.KubeletConfiguration, error) {
	resp := pollConfigz(ctx, 5*time.Minute, 5*time.Second, nodeName, namespace, useProxy, standaloneMode)
	if len(resp) == 0 {
		return nil, fmt.Errorf("failed to fetch /configz from %q", nodeName)
	}
	kubeCfg, err := decodeConfigz(resp)
	if err != nil {
		return nil, err
	}
	return kubeCfg, nil
}

// returns a status 200 response from the /configz endpoint or nil if fails
func pollConfigz(ctx context.Context, timeout time.Duration, pollInterval time.Duration, nodeName, namespace string, useProxy bool, standaloneMode bool) []byte {
	endpoint := ""
	if useProxy {
		// start local proxy, so we can send graceful deletion over query string, rather than body parameter
		framework.Logf("Opening proxy to cluster")
		tk := e2ekubectl.NewTestKubeconfig(framework.TestContext.CertDir, framework.TestContext.Host, framework.TestContext.KubeConfig, framework.TestContext.KubeContext, framework.TestContext.KubectlPath, namespace)
		cmd := tk.KubectlCmd("proxy", "-p", "0")
		stdout, stderr, err := framework.StartCmdAndStreamOutput(cmd)
		framework.ExpectNoError(err)
		defer stdout.Close()
		defer stderr.Close()
		defer framework.TryKill(cmd)

		buf := make([]byte, 128)
		var n int
		n, err = stdout.Read(buf)
		framework.ExpectNoError(err)
		output := string(buf[:n])
		proxyRegexp := regexp.MustCompile("Starting to serve on 127.0.0.1:([0-9]+)")
		match := proxyRegexp.FindStringSubmatch(output)
		gomega.Expect(match).To(gomega.HaveLen(2))
		port, err := strconv.Atoi(match[1])
		framework.ExpectNoError(err)
		framework.Logf("http requesting node kubelet /configz")
		endpoint = fmt.Sprintf("http://127.0.0.1:%d/api/v1/nodes/%s/proxy/configz", port, nodeName)
	} else if !standaloneMode {
		endpoint = fmt.Sprintf("%s/api/v1/nodes/%s/proxy/configz", framework.TestContext.Host, framework.TestContext.NodeName)
	} else {
		endpoint = fmt.Sprintf("https://127.0.0.1:%d/configz", ports.KubeletPort)
	}
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
	client := &http.Client{Transport: tr}
	req, err := http.NewRequest("GET", endpoint, nil)
	framework.ExpectNoError(err)
	if !useProxy {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", framework.TestContext.BearerToken))
	}
	req.Header.Add("Accept", "application/json")

	var respBody []byte
	err = wait.PollUntilContextTimeout(ctx, pollInterval, timeout, true, func(ctx context.Context) (bool, error) {
		resp, err := client.Do(req)
		if err != nil {
			framework.Logf("Failed to get /configz, retrying. Error: %v", err)
			return false, nil
		}
		defer resp.Body.Close()

		if resp.StatusCode != 200 {
			framework.Logf("/configz response status not 200, retrying. Response was: %+v", resp)
			return false, nil
		}

		respBody, err = io.ReadAll(resp.Body)
		if err != nil {
			framework.Logf("failed to read body from /configz response, retrying. Error: %v", err)
			return false, nil
		}

		return true, nil
	})
	framework.ExpectNoError(err, "Failed to get successful response from /configz")

	return respBody
}

// Decodes the http response from /configz and returns a kubeletconfig.KubeletConfiguration (internal type).
func decodeConfigz(respBody []byte) (*kubeletconfig.KubeletConfiguration, error) {
	// This hack because /configz reports the following structure:
	// {"kubeletconfig": {the JSON representation of kubeletconfigv1beta1.KubeletConfiguration}}
	type configzWrapper struct {
		ComponentConfig kubeletconfigv1beta1.KubeletConfiguration `json:"kubeletconfig"`
	}

	configz := configzWrapper{}
	kubeCfg := kubeletconfig.KubeletConfiguration{}

	err := json.Unmarshal(respBody, &configz)
	if err != nil {
		return nil, err
	}

	scheme, _, err := kubeletconfigscheme.NewSchemeAndCodecs()
	if err != nil {
		return nil, err
	}
	err = scheme.Convert(&configz.ComponentConfig, &kubeCfg, nil)
	if err != nil {
		return nil, err
	}

	return &kubeCfg, nil
}
