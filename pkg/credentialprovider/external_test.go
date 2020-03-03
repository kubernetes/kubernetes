/*
Copyright 2020 The Kubernetes Authors.

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

package credentialprovider

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/kubernetes/pkg/credentialprovider/apis/registrycredentials"
)

func TestConfigDeserialization(t *testing.T) {
	tests := []struct {
		configYaml    []byte
		expectedObj   *registrycredentials.RegistryCredentialConfig
		matchExpected bool
	}{
		{
			configYaml: []byte(`apiVersion: registrycredentials.k8s.io/v1alpha1
kind: RegistryCredentialConfig
providers:
-
  imageMatchers:
  - "*.dkr.ecr.*.amazonaws.com"
  - "*.dkr.ecr.*.amazonaws.com.cn"
  exec:
    command: ecr-creds
    args:
    - token
    env:
    - name: XYZ
      value: envvalue
    apiVersion: registrycredentials.k8s.io/v1alpha1`),
			expectedObj: &registrycredentials.RegistryCredentialConfig{
				Providers: []registrycredentials.RegistryCredentialProvider{
					registrycredentials.RegistryCredentialProvider{
						ImageMatchers: []string{
							"*.dkr.ecr.*.amazonaws.com",
							"*.dkr.ecr.*.amazonaws.com.cn",
						},
						Exec: registrycredentials.ExecConfig{
							Command: "ecr-creds",
							Args:    []string{"token"},
							Env: []registrycredentials.ExecEnvVar{
								registrycredentials.ExecEnvVar{
									Name:  "XYZ",
									Value: "envvalue",
								},
							},
						},
					},
				},
			},
			matchExpected: true,
		},
	}
	for _, test := range tests {
		actualObj, err := decode(test.configYaml)
		if err != nil {
			t.Errorf("Decode failed with error: %s", err.Error())
		}

		if diff := cmp.Diff(test.expectedObj, actualObj); diff != "" {
			t.Errorf("Unexpected diff (-want +got):\n%s", diff)
		}
	}
}
