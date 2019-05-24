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
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog"
	"sigs.k8s.io/yaml"
)

const (
	secretNamespace      = "kube-system"
	secretCloudConfigKey = "cloud-config"
)

// The configure type for Azure cloud provider secret. Supported values are:
// * all            : configure applied for components (kubelet and controller-manager). This is the default value.
// * node           : configure applied for nodes (kubelet).
// * control-plane  : configure applied for control plane components (controller-manager).
//
// For different configure types, the secret name would also be different:
// * all            : secret name would be azure-cloud-provider.
// * node           : secret name would azure-cloud-provider-node.
// * control-plane  : secret name would be azure-cloud-provider-control-plane.
type secretConfigureType string

const (
	secretConfigureAll          secretConfigureType = "all"
	secretConfigureNode         secretConfigureType = "node"
	secretConfigureControlPlane secretConfigureType = "control-plane"
)

// The override type for Azure cloud provider secret. Supported values are:
// * no   : The values from secret won't override any configures from local cloud-config file.
// * must : The values from secret would override all configures from local cloud-config file.
// * can  : The values from secret would override only configurations that are explicitly set in the secret. This is the default value.
type secretOverrideType string

const (
	secretOverrideTypeNo   secretOverrideType = "no"
	secretOverrideTypeMust secretOverrideType = "must"
	secretOverrideTypeCan  secretOverrideType = "can"
)

func (az *Cloud) initializeCloudFromSecret() {
	config, err := az.getConfigFromSecret()
	if err != nil {
		klog.Warningf("Failed to get cloud-config from secret: %v, skip initializing from secret", err)
		return
	}

	if config == nil {
		// Skip re-initialization if the config is not override.
		return
	}

	if err := az.initializeCloudFromConfig(config, true); err != nil {
		klog.Errorf("Failed to initialize Azure cloud provider: %v", err)
	}
}

func (az *Cloud) getConfigFromSecret() (*Config, error) {
	// No override, return nil.
	if az.Config.OverrideType == secretOverrideTypeNo {
		return nil, nil
	}

	secretName := getConfigSecretName(az.Config.ConfigType)
	secret, err := az.kubeClient.CoreV1().Secrets(secretNamespace).Get(secretName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("Failed to get secret %s: %v", secretName, err)
	}

	cloudConfigData, ok := secret.Data[secretCloudConfigKey]
	if !ok {
		return nil, fmt.Errorf("cloud-config is not set in the secret (%s)", secretName)
	}

	config := Config{}
	if az.Config.OverrideType == "" || az.Config.OverrideType == secretOverrideTypeCan {
		// "can" override, set default value to existing config.
		config = az.Config
	}

	err = yaml.Unmarshal(cloudConfigData, &config)
	if err != nil {
		return nil, fmt.Errorf("Failed to parse Azure cloud-config: %v", err)
	}

	return &config, nil
}

func getConfigSecretName(configType secretConfigureType) string {
	switch configType {
	case secretConfigureAll:
		return azureSecretNamePrefix
	case secretConfigureNode:
		return fmt.Sprintf("%s-node", azureSecretNamePrefix)
	case secretConfigureControlPlane:
		return fmt.Sprintf("%s-control-plane", azureSecretNamePrefix)

	default:
		// default secret name is azure-cloud-provider.
		return azureSecretNamePrefix
	}
}
