// +build !providerless

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
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"sigs.k8s.io/yaml"
)

const (
	cloudConfigNamespace  = "kube-system"
	cloudConfigKey        = "cloud-config"
	cloudConfigSecretName = "azure-cloud-provider"
)

// The config type for Azure cloud provider secret. Supported values are:
// * file   : The values are read from local cloud-config file.
// * secret : The values from secret would override all configures from local cloud-config file.
// * merge  : The values from secret would override only configurations that are explicitly set in the secret. This is the default value.
type cloudConfigType string

const (
	cloudConfigTypeFile   cloudConfigType = "file"
	cloudConfigTypeSecret cloudConfigType = "secret"
	cloudConfigTypeMerge  cloudConfigType = "merge"
)

// InitializeCloudFromSecret initializes Azure cloud provider from Kubernetes secret.
func (az *Cloud) InitializeCloudFromSecret() {
	config, err := az.getConfigFromSecret()
	if err != nil {
		klog.Warningf("Failed to get cloud-config from secret: %v, skip initializing from secret", err)
		return
	}

	if config == nil {
		// Skip re-initialization if the config is not override.
		return
	}

	if err := az.InitializeCloudFromConfig(config, true); err != nil {
		klog.Errorf("Failed to initialize Azure cloud provider: %v", err)
	}
}

func (az *Cloud) getConfigFromSecret() (*Config, error) {
	// Read config from file and no override, return nil.
	if az.Config.CloudConfigType == cloudConfigTypeFile {
		return nil, nil
	}

	secret, err := az.KubeClient.CoreV1().Secrets(cloudConfigNamespace).Get(context.TODO(), cloudConfigSecretName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get secret %s: %v", cloudConfigSecretName, err)
	}

	cloudConfigData, ok := secret.Data[cloudConfigKey]
	if !ok {
		return nil, fmt.Errorf("cloud-config is not set in the secret (%s)", cloudConfigSecretName)
	}

	config := Config{}
	if az.Config.CloudConfigType == "" || az.Config.CloudConfigType == cloudConfigTypeMerge {
		// Merge cloud config, set default value to existing config.
		config = az.Config
	}

	err = yaml.Unmarshal(cloudConfigData, &config)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Azure cloud-config: %v", err)
	}

	return &config, nil
}
