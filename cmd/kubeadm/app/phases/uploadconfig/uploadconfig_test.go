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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestUploadConfiguration(t *testing.T) {
	tests := []struct {
		name    string
		cfg     *kubeadmapi.MasterConfiguration
		wantErr bool
	}{
		{
			"basic validation with correct key",
			&kubeadmapi.MasterConfiguration{
				KubernetesVersion: "1.7.3",
			},
			false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			if err := UploadConfiguration(tt.cfg, client); (err != nil) != tt.wantErr {
				t.Errorf("UploadConfiguration() error = %v, wantErr %v", err, tt.wantErr)
			}
			masterCfg, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(kubeadmconstants.MasterConfigurationConfigMap, metav1.GetOptions{})
			if err != nil {
				t.Errorf("Fail to query ConfigMap error = %v", err)
			} else if masterCfg.Data[kubeadmconstants.MasterConfigurationConfigMapKey] == "" {
				t.Errorf("Fail to find ConfigMap key = %v", err)
			}
		})
	}
}
