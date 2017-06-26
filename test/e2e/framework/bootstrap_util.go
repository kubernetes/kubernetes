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

package framework

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

func WaitforSignedBootStrapToken(c clientset.Interface, tokenID string) error {

	return wait.Poll(Poll, 2*time.Minute, func() (bool, error) {
		cfgMap, err := c.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
		if err != nil {
			Failf("Failed to get cluster-info configMap: %v", err)
			return false, err
		}
		_, ok := cfgMap.Data[bootstrapapi.JWSSignatureKeyPrefix+tokenID]
		if !ok {
			return false, nil
		}
		return true, nil
	})
}

func WaitForSignedBootstrapTokenToGetUpdated(c clientset.Interface, tokenID string, signedToken string) error {

	return wait.Poll(Poll, 2*time.Minute, func() (bool, error) {
		cfgMap, err := c.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
		if err != nil {
			Failf("Failed to get cluster-info configMap: %v", err)
			return false, err
		}
		updated, ok := cfgMap.Data[bootstrapapi.JWSSignatureKeyPrefix+tokenID]
		if !ok || updated == signedToken {
			return false, nil
		}
		return true, nil
	})
}

func WaitForSignedBootstrapTokenToDisappear(c clientset.Interface, tokenID string) error {

	return wait.Poll(Poll, 2*time.Minute, func() (bool, error) {
		cfgMap, err := c.CoreV1().ConfigMaps(metav1.NamespacePublic).Get(bootstrapapi.ConfigMapClusterInfo, metav1.GetOptions{})
		if err != nil {
			Failf("Failed to get cluster-info configMap: %v", err)
			return false, err
		}
		_, ok := cfgMap.Data[bootstrapapi.JWSSignatureKeyPrefix+tokenID]
		if ok {
			return false, nil
		}
		return true, nil
	})
}
