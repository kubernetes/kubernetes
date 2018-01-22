/*
Copyright 2018 The Kubernetes Authors.

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

package config

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	// create a place holder file to hold per test config
	configFile, err := ioutil.TempFile("", "admission-config")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	configFileName := configFile.Name()
	defer os.Remove(configFileName)

	if err = configFile.Close(); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	// individual test scenarios
	testCases := []struct {
		CaseName   string
		ConfigBody string
		ExpecteErr bool
	}{
		// valid configuration
		{
			CaseName: "valid configuration",
			ConfigBody: `
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: WebhookAdmission
metadata:
  name: webhook-config
kubeConfigFile: /var/run/kubernetes/webhook.kubeconfig
`,
			ExpecteErr: false,
		},

		// invalid configuration: kubeConfigFile should be with absolute path
		{
			CaseName: "invalid configuration kubeConfigFile",
			ConfigBody: `
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: WebhookAdmission
metadata:
  name: webhook-config
kubeConfigFile: webhook.kubeconfig
`,
			ExpecteErr: true,
		},

		// invalid configuration kind
		{
			CaseName: "invalid configuration",
			ConfigBody: `
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: InvalidWebhookAdmission
metadata:
  name: webhook-config
kubeConfigFile: /var/run/kubernetes/webhook.kubeconfig
`,
			ExpecteErr: true,
		},
	}

	for _, testcase := range testCases {
		func() {
			if err = ioutil.WriteFile(configFileName, []byte(testcase.ConfigBody), 0644); err != nil {
				t.Fatalf("unexpected err writing temp file: %v", err)
			}

			configFile, err := os.Open(configFileName)
			if err != nil {
				t.Fatalf("failed to read test config: %v", err)
			}
			defer configFile.Close()

			_, err = LoadConfig(configFile)
			if testcase.ExpecteErr && err == nil {
				t.Errorf("expect error but got none")
			}
			if !testcase.ExpecteErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		}()
	}
}
