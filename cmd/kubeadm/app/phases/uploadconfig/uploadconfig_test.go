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

package uploadconfig

import (
	"context"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientsetfake "k8s.io/client-go/kubernetes/fake"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

func TestUploadConfiguration(t *testing.T) {
	tests := []struct {
		name           string
		updateExisting bool
		verifyResult   bool
	}{
		{
			name:         "basic validation with correct key",
			verifyResult: true,
		},
		{
			name:           "update existing should report no error",
			updateExisting: true,
			verifyResult:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t2 *testing.T) {
			cfg, err := configutil.DefaultedStaticInitConfiguration()
			if err != nil {
				t2.Fatalf("UploadConfiguration() error = %v", err)
			}
			cfg.ComponentConfigs = kubeadmapi.ComponentConfigMap{}
			cfg.ClusterConfiguration.KubernetesVersion = kubeadmconstants.MinimumControlPlaneVersion.WithPatch(10).String()
			cfg.NodeRegistration.Name = "node-foo"
			cfg.NodeRegistration.CRISocket = kubeadmconstants.DefaultCRISocket

			client := clientsetfake.NewSimpleClientset()
			// For idempotent test, we check the result of the second call.
			if err := UploadConfiguration(cfg, client); err != nil {
				t2.Fatalf("UploadConfiguration() error = %v", err)
			}
			if tt.updateExisting {
				if err := UploadConfiguration(cfg, client); err != nil {
					t2.Fatalf("UploadConfiguration() error = %v", err)
				}
			}
			if tt.verifyResult {
				controlPlaneCfg, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(context.TODO(), kubeadmconstants.KubeadmConfigConfigMap, metav1.GetOptions{})
				if err != nil {
					t2.Fatalf("Fail to query ConfigMap error = %v", err)
				}
				configData := controlPlaneCfg.Data[kubeadmconstants.ClusterConfigurationConfigMapKey]
				if configData == "" {
					t2.Fatal("Fail to find ClusterConfigurationConfigMapKey key")
				}

				decodedCfg := &kubeadmapi.ClusterConfiguration{}
				if err := runtime.DecodeInto(kubeadmscheme.Codecs.UniversalDecoder(), []byte(configData), decodedCfg); err != nil {
					t2.Fatalf("unable to decode config from bytes: %v", err)
				}

				if len(decodedCfg.ComponentConfigs) != 0 {
					t2.Errorf("unexpected component configs in decodedCfg: %d", len(decodedCfg.ComponentConfigs))
				}

				// Force initialize with an empty map so that reflect.DeepEqual works
				decodedCfg.ComponentConfigs = kubeadmapi.ComponentConfigMap{}

				if !reflect.DeepEqual(decodedCfg, &cfg.ClusterConfiguration) {
					t2.Errorf("the initial and decoded ClusterConfiguration didn't match:\n%#v\n===\n%#v", decodedCfg, &cfg.ClusterConfiguration)
				}
			}
		})
	}
}
