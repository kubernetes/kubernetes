/*
Copyright 2019 The Kubernetes Authors.

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

package azure

import (
	"testing"

	"github.com/Azure/go-autorest/autorest/to"
	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	"k8s.io/legacy-cloud-providers/azure/auth"
	"sigs.k8s.io/yaml"
)

func getTestConfig() *Config {
	return &Config{
		AzureAuthConfig: auth.AzureAuthConfig{
			TenantID:        "TenantID",
			SubscriptionID:  "SubscriptionID",
			AADClientID:     "AADClientID",
			AADClientSecret: "AADClientSecret",
		},
		ResourceGroup:               "ResourceGroup",
		RouteTableName:              "RouteTableName",
		RouteTableResourceGroup:     "RouteTableResourceGroup",
		Location:                    "Location",
		SubnetName:                  "SubnetName",
		VnetName:                    "VnetName",
		PrimaryAvailabilitySetName:  "PrimaryAvailabilitySetName",
		PrimaryScaleSetName:         "PrimaryScaleSetName",
		LoadBalancerSku:             "LoadBalancerSku",
		ExcludeMasterFromStandardLB: to.BoolPtr(true),
	}
}

func getTestCloudConfigTypeSecretConfig() *Config {
	return &Config{
		AzureAuthConfig: auth.AzureAuthConfig{
			TenantID:       "TenantID",
			SubscriptionID: "SubscriptionID",
		},
		ResourceGroup:           "ResourceGroup",
		RouteTableName:          "RouteTableName",
		RouteTableResourceGroup: "RouteTableResourceGroup",
		SecurityGroupName:       "SecurityGroupName",
		CloudConfigType:         cloudConfigTypeSecret,
	}
}

func getTestCloudConfigTypeMergeConfig() *Config {
	return &Config{
		AzureAuthConfig: auth.AzureAuthConfig{
			TenantID:       "TenantID",
			SubscriptionID: "SubscriptionID",
		},
		ResourceGroup:           "ResourceGroup",
		RouteTableName:          "RouteTableName",
		RouteTableResourceGroup: "RouteTableResourceGroup",
		SecurityGroupName:       "SecurityGroupName",
		CloudConfigType:         cloudConfigTypeMerge,
	}
}

func getTestCloudConfigTypeMergeConfigExpected() *Config {
	config := getTestConfig()
	config.SecurityGroupName = "SecurityGroupName"
	config.CloudConfigType = cloudConfigTypeMerge
	return config
}

func TestGetConfigFromSecret(t *testing.T) {
	emptyConfig := &Config{}
	tests := []struct {
		name           string
		existingConfig *Config
		secretConfig   *Config
		expected       *Config
		expectErr      bool
	}{
		{
			name: "Azure config shouldn't be override when cloud config type is file",
			existingConfig: &Config{
				ResourceGroup:   "ResourceGroup1",
				CloudConfigType: cloudConfigTypeFile,
			},
			secretConfig: getTestConfig(),
			expected:     nil,
		},
		{
			name:           "Azure config should be override when cloud config type is secret",
			existingConfig: getTestCloudConfigTypeSecretConfig(),
			secretConfig:   getTestConfig(),
			expected:       getTestConfig(),
		},
		{
			name:           "Azure config should be override when cloud config type is merge",
			existingConfig: getTestCloudConfigTypeMergeConfig(),
			secretConfig:   getTestConfig(),
			expected:       getTestCloudConfigTypeMergeConfigExpected(),
		},
		{
			name:           "Error should be reported when secret doesn't exists",
			existingConfig: getTestCloudConfigTypeMergeConfig(),
			expectErr:      true,
		},
		{
			name:           "Error should be reported when secret exists but cloud-config data is not provided",
			existingConfig: getTestCloudConfigTypeMergeConfig(),
			secretConfig:   emptyConfig,
			expectErr:      true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			az := &Cloud{
				kubeClient: fakeclient.NewSimpleClientset(),
			}
			if test.existingConfig != nil {
				az.Config = *test.existingConfig
			}
			if test.secretConfig != nil {
				secret := &v1.Secret{
					Type: v1.SecretTypeOpaque,
					ObjectMeta: metav1.ObjectMeta{
						Name:      "azure-cloud-provider",
						Namespace: "kube-system",
					},
				}
				if test.secretConfig != emptyConfig {
					secretData, err := yaml.Marshal(test.secretConfig)
					assert.NoError(t, err, test.name)
					secret.Data = map[string][]byte{
						"cloud-config": secretData,
					}
				}
				_, err := az.kubeClient.CoreV1().Secrets(cloudConfigNamespace).Create(secret)
				assert.NoError(t, err, test.name)
			}

			real, err := az.getConfigFromSecret()
			if test.expectErr {
				assert.Error(t, err, test.name)
				return
			}

			assert.NoError(t, err, test.name)
			assert.Equal(t, test.expected, real, test.name)
		})
	}
}
