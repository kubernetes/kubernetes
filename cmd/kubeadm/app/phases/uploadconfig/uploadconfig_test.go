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
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

func TestUploadConfiguration(t *testing.T) {
	tests := []struct {
		name           string
		errOnCreate    error
		errOnUpdate    error
		updateExisting bool
		errExpected    bool
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
		{
			name:        "unexpected errors for create should be returned",
			errOnCreate: apierrors.NewUnauthorized(""),
			errExpected: true,
		},
		{
			name:           "update existing show report error if unexpected error for update is returned",
			errOnUpdate:    apierrors.NewUnauthorized(""),
			updateExisting: true,
			errExpected:    true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t2 *testing.T) {
			initialcfg := &kubeadmapiv1beta2.InitConfiguration{
				LocalAPIEndpoint: kubeadmapiv1beta2.APIEndpoint{
					AdvertiseAddress: "1.2.3.4",
				},
				BootstrapTokens: []kubeadmapiv1beta2.BootstrapToken{
					{
						Token: &kubeadmapiv1beta2.BootstrapTokenString{
							ID:     "abcdef",
							Secret: "abcdef0123456789",
						},
					},
				},
				NodeRegistration: kubeadmapiv1beta2.NodeRegistrationOptions{
					Name:      "node-foo",
					CRISocket: "/var/run/custom-cri.sock",
				},
			}
			clustercfg := &kubeadmapiv1beta2.ClusterConfiguration{
				KubernetesVersion: kubeadmconstants.MinimumControlPlaneVersion.WithPatch(10).String(),
			}
			cfg, err := configutil.DefaultedInitConfiguration(initialcfg, clustercfg)

			if err != nil {
				t2.Fatalf("UploadConfiguration() error = %v", err)
			}

			cfg.ComponentConfigs = kubeadmapi.ComponentConfigMap{}

			status := &kubeadmapi.ClusterStatus{
				APIEndpoints: map[string]kubeadmapi.APIEndpoint{
					"node-foo": cfg.LocalAPIEndpoint,
				},
			}

			client := clientsetfake.NewSimpleClientset()
			if tt.errOnCreate != nil {
				client.PrependReactor("create", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
					return true, nil, tt.errOnCreate
				})
			}
			// For idempotent test, we check the result of the second call.
			if err := UploadConfiguration(cfg, client); !tt.updateExisting && (err != nil) != tt.errExpected {
				t2.Fatalf("UploadConfiguration() error = %v, wantErr %v", err, tt.errExpected)
			}
			if tt.updateExisting {
				if tt.errOnUpdate != nil {
					client.PrependReactor("update", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
						return true, nil, tt.errOnUpdate
					})
				}
				if err := UploadConfiguration(cfg, client); (err != nil) != tt.errExpected {
					t2.Fatalf("UploadConfiguration() error = %v", err)
				}
			}
			if tt.verifyResult {
				controlPlaneCfg, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(kubeadmconstants.KubeadmConfigConfigMap, metav1.GetOptions{})
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
					t2.Errorf("the initial and decoded ClusterConfiguration didn't match:\n%t\n===\n%t", decodedCfg.ComponentConfigs == nil, cfg.ComponentConfigs == nil)
				}

				statusData := controlPlaneCfg.Data[kubeadmconstants.ClusterStatusConfigMapKey]
				if statusData == "" {
					t2.Fatal("failed to find ClusterStatusConfigMapKey key")
				}

				decodedStatus := &kubeadmapi.ClusterStatus{}
				if err := runtime.DecodeInto(kubeadmscheme.Codecs.UniversalDecoder(), []byte(statusData), decodedStatus); err != nil {
					t2.Fatalf("unable to decode status from bytes: %v", err)
				}

				if !reflect.DeepEqual(decodedStatus, status) {
					t2.Error("the initial and decoded ClusterStatus didn't match")
				}
			}
		})
	}
}

func TestMutateClusterStatus(t *testing.T) {
	cm := &v1.ConfigMap{
		Data: map[string]string{
			kubeadmconstants.ClusterStatusConfigMapKey: "",
		},
	}

	endpoints := map[string]kubeadmapi.APIEndpoint{
		"some-node": {
			AdvertiseAddress: "127.0.0.1",
			BindPort:         6443,
		},
	}

	err := mutateClusterStatus(cm, func(cs *kubeadmapi.ClusterStatus) error {
		cs.APIEndpoints = endpoints
		return nil
	})
	if err != nil {
		t.Fatalf("could not mutate cluster status: %v", err)
	}

	// Try to unmarshal the cluster status back and compare with the original mutated structure
	cs, err := configutil.UnmarshalClusterStatus(cm.Data)
	if err != nil {
		t.Fatalf("could not unmarshal cluster status: %v", err)
	}

	if !reflect.DeepEqual(cs.APIEndpoints, endpoints) {
		t.Fatalf("mutation of cluster status failed: %v", err)
	}
}
