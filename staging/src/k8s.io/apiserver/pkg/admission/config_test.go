/*
Copyright 2017 The Kubernetes Authors.

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

package admission

import (
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apiserver/pkg/apis/apiserver"
	apiserverapiv1 "k8s.io/apiserver/pkg/apis/apiserver/v1"
	apiserverapiv1alpha1 "k8s.io/apiserver/pkg/apis/apiserver/v1alpha1"
)

func TestReadAdmissionConfiguration(t *testing.T) {
	// create a place holder file to hold per test config
	configFile, err := ioutil.TempFile("", "admission-plugin-config")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	defer os.Remove(configFile.Name())
	if err = configFile.Close(); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	configFileName := configFile.Name()
	// the location that will be fixed up to be relative to the test config file.
	imagePolicyWebhookFile, err := makeAbs("image-policy-webhook.json", os.TempDir())
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	// individual test scenarios
	testCases := map[string]struct {
		ConfigBody              string
		ExpectedAdmissionConfig *apiserver.AdmissionConfiguration
		PluginNames             []string
	}{
		"v1alpha1 configuration - path fixup": {
			ConfigBody: `{
"apiVersion": "apiserver.k8s.io/v1alpha1",
"kind": "AdmissionConfiguration",
"plugins": [
  {"name": "ImagePolicyWebhook", "path": "image-policy-webhook.json"},
  {"name": "ResourceQuota"}
]}`,
			ExpectedAdmissionConfig: &apiserver.AdmissionConfiguration{
				Plugins: []apiserver.AdmissionPluginConfiguration{
					{
						Name: "ImagePolicyWebhook",
						Path: imagePolicyWebhookFile,
					},
					{
						Name: "ResourceQuota",
					},
				},
			},
			PluginNames: []string{},
		},
		"v1alpha1 configuration - abspath": {
			ConfigBody: `{
"apiVersion": "apiserver.k8s.io/v1alpha1",
"kind": "AdmissionConfiguration",
"plugins": [
  {"name": "ImagePolicyWebhook", "path": "/tmp/image-policy-webhook.json"},
  {"name": "ResourceQuota"}
]}`,
			ExpectedAdmissionConfig: &apiserver.AdmissionConfiguration{
				Plugins: []apiserver.AdmissionPluginConfiguration{
					{
						Name: "ImagePolicyWebhook",
						Path: "/tmp/image-policy-webhook.json",
					},
					{
						Name: "ResourceQuota",
					},
				},
			},
			PluginNames: []string{},
		},
		"v1 configuration - path fixup": {
			ConfigBody: `{
"apiVersion": "apiserver.config.k8s.io/v1",
"kind": "AdmissionConfiguration",
"plugins": [
  {"name": "ImagePolicyWebhook", "path": "image-policy-webhook.json"},
  {"name": "ResourceQuota"}
]}`,
			ExpectedAdmissionConfig: &apiserver.AdmissionConfiguration{
				Plugins: []apiserver.AdmissionPluginConfiguration{
					{
						Name: "ImagePolicyWebhook",
						Path: imagePolicyWebhookFile,
					},
					{
						Name: "ResourceQuota",
					},
				},
			},
			PluginNames: []string{},
		},
		"v1 configuration - abspath": {
			ConfigBody: `{
"apiVersion": "apiserver.config.k8s.io/v1",
"kind": "AdmissionConfiguration",
"plugins": [
  {"name": "ImagePolicyWebhook", "path": "/tmp/image-policy-webhook.json"},
  {"name": "ResourceQuota"}
]}`,
			ExpectedAdmissionConfig: &apiserver.AdmissionConfiguration{
				Plugins: []apiserver.AdmissionPluginConfiguration{
					{
						Name: "ImagePolicyWebhook",
						Path: "/tmp/image-policy-webhook.json",
					},
					{
						Name: "ResourceQuota",
					},
				},
			},
			PluginNames: []string{},
		},
		"legacy configuration with using legacy plugins": {
			ConfigBody: `{
"imagePolicy": {
  "kubeConfigFile": "/home/user/.kube/config",
  "allowTTL": 30,
  "denyTTL": 30,
  "retryBackoff": 500,
  "defaultAllow": true
},
"podNodeSelectorPluginConfig": {
  "clusterDefaultNodeSelector": ""
}  
}`,
			ExpectedAdmissionConfig: &apiserver.AdmissionConfiguration{
				Plugins: []apiserver.AdmissionPluginConfiguration{
					{
						Name: "ImagePolicyWebhook",
						Path: configFileName,
					},
					{
						Name: "PodNodeSelector",
						Path: configFileName,
					},
				},
			},
			PluginNames: []string{"ImagePolicyWebhook", "PodNodeSelector"},
		},
		"legacy configuration not using legacy plugins": {
			ConfigBody: `{
"imagePolicy": {
  "kubeConfigFile": "/home/user/.kube/config",
  "allowTTL": 30,
  "denyTTL": 30,
  "retryBackoff": 500,
  "defaultAllow": true
},
"podNodeSelectorPluginConfig": {
  "clusterDefaultNodeSelector": ""
}  
}`,
			ExpectedAdmissionConfig: &apiserver.AdmissionConfiguration{},
			PluginNames:             []string{"NamespaceLifecycle", "InitialResources"},
		},
	}

	scheme := runtime.NewScheme()
	require.NoError(t, apiserver.AddToScheme(scheme))
	require.NoError(t, apiserverapiv1alpha1.AddToScheme(scheme))
	require.NoError(t, apiserverapiv1.AddToScheme(scheme))

	for testName, testCase := range testCases {
		if err = ioutil.WriteFile(configFileName, []byte(testCase.ConfigBody), 0644); err != nil {
			t.Fatalf("unexpected err writing temp file: %v", err)
		}
		config, err := ReadAdmissionConfiguration(testCase.PluginNames, configFileName, scheme)
		if err != nil {
			t.Fatalf("unexpected err: %v", err)
		}
		if !reflect.DeepEqual(config.(configProvider).config, testCase.ExpectedAdmissionConfig) {
			t.Errorf("%s: Expected:\n\t%#v\nGot:\n\t%#v", testName, testCase.ExpectedAdmissionConfig, config.(configProvider).config)
		}
	}
}

func TestEmbeddedConfiguration(t *testing.T) {
	// create a place holder file to hold per test config
	configFile, err := ioutil.TempFile("", "admission-plugin-config")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	defer os.Remove(configFile.Name())
	if err = configFile.Close(); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	configFileName := configFile.Name()

	testCases := map[string]struct {
		ConfigBody     string
		ExpectedConfig string
	}{
		"v1alpha1 versioned configuration": {
			ConfigBody: `{
				"apiVersion": "apiserver.k8s.io/v1alpha1",
				"kind": "AdmissionConfiguration",
				"plugins": [
				  {
					"name": "Foo",
					"configuration": {
					  "apiVersion": "foo.admission.k8s.io/v1alpha1",
					  "kind": "Configuration",
					  "foo": "bar"
					}
				  }
				]}`,
			ExpectedConfig: `{
			  "apiVersion": "foo.admission.k8s.io/v1alpha1",
			  "kind": "Configuration",
			  "foo": "bar"
			}`,
		},
		"v1alpha1 legacy configuration": {
			ConfigBody: `{
				"apiVersion": "apiserver.k8s.io/v1alpha1",
				"kind": "AdmissionConfiguration",
				"plugins": [
				  {
					"name": "Foo",
					"configuration": {
					  "foo": "bar"
					}
				  }
				]}`,
			ExpectedConfig: `{
			  "foo": "bar"
			}`,
		},
		"v1 versioned configuration": {
			ConfigBody: `{
				"apiVersion": "apiserver.config.k8s.io/v1",
				"kind": "AdmissionConfiguration",
				"plugins": [
				  {
					"name": "Foo",
					"configuration": {
					  "apiVersion": "foo.admission.k8s.io/v1alpha1",
					  "kind": "Configuration",
					  "foo": "bar"
					}
				  }
				]}`,
			ExpectedConfig: `{
			  "apiVersion": "foo.admission.k8s.io/v1alpha1",
			  "kind": "Configuration",
			  "foo": "bar"
			}`,
		},
		"v1 legacy configuration": {
			ConfigBody: `{
				"apiVersion": "apiserver.config.k8s.io/v1",
				"kind": "AdmissionConfiguration",
				"plugins": [
				  {
					"name": "Foo",
					"configuration": {
					  "foo": "bar"
					}
				  }
				]}`,
			ExpectedConfig: `{
			  "foo": "bar"
			}`,
		},
	}

	for desc, test := range testCases {
		scheme := runtime.NewScheme()
		require.NoError(t, apiserver.AddToScheme(scheme))
		require.NoError(t, apiserverapiv1alpha1.AddToScheme(scheme))
		require.NoError(t, apiserverapiv1.AddToScheme(scheme))

		if err = ioutil.WriteFile(configFileName, []byte(test.ConfigBody), 0644); err != nil {
			t.Errorf("[%s] unexpected err writing temp file: %v", desc, err)
			continue
		}
		config, err := ReadAdmissionConfiguration([]string{"Foo"}, configFileName, scheme)
		if err != nil {
			t.Errorf("[%s] unexpected err: %v", desc, err)
			continue
		}
		r, err := config.ConfigFor("Foo")
		if err != nil {
			t.Errorf("[%s] Failed to get Foo config: %v", desc, err)
			continue
		}
		bs, err := ioutil.ReadAll(r)
		if err != nil {
			t.Errorf("[%s] Failed to read Foo config data: %v", desc, err)
			continue
		}

		if !equalJSON(test.ExpectedConfig, string(bs)) {
			t.Errorf("Unexpected config: expected=%q got=%q", test.ExpectedConfig, string(bs))
		}
	}
}

func equalJSON(a, b string) bool {
	var x, y interface{}
	if err := json.Unmarshal([]byte(a), &x); err != nil {
		return false
	}
	if err := json.Unmarshal([]byte(b), &y); err != nil {
		return false
	}
	return reflect.DeepEqual(x, y)
}
