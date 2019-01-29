/*
Copyright 2015 The Kubernetes Authors.

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

// This test is volumes test for configmap.
package storage

import (
	. "github.com/onsi/ginkgo"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// These tests need privileged containers, which are disabled by default.
var _ = utils.SIGDescribe("Volumes", func() {
	f := framework.NewDefaultFramework("volume")

	// note that namespace deletion is handled by delete-namespace flag
	// filled inside BeforeEach
	var cs clientset.Interface
	var namespace *v1.Namespace

	BeforeEach(func() {
		cs = f.ClientSet
		namespace = f.Namespace
	})

	Describe("ConfigMap", func() {
		It("should be mountable", func() {
			config := framework.VolumeTestConfig{
				Namespace: namespace.Name,
				Prefix:    "configmap",
			}

			defer framework.VolumeTestCleanup(f, config)
			configMap := &v1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ConfigMap",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: config.Prefix + "-map",
				},
				Data: map[string]string{
					"first":  "this is the first file",
					"second": "this is the second file",
					"third":  "this is the third file",
				},
			}
			if _, err := cs.CoreV1().ConfigMaps(namespace.Name).Create(configMap); err != nil {
				framework.Failf("unable to create test configmap: %v", err)
			}
			defer func() {
				_ = cs.CoreV1().ConfigMaps(namespace.Name).Delete(configMap.Name, nil)
			}()

			// Test one ConfigMap mounted several times to test #28502
			tests := []framework.VolumeTest{
				{
					Volume: v1.VolumeSource{
						ConfigMap: &v1.ConfigMapVolumeSource{
							LocalObjectReference: v1.LocalObjectReference{
								Name: config.Prefix + "-map",
							},
							Items: []v1.KeyToPath{
								{
									Key:  "first",
									Path: "firstfile",
								},
							},
						},
					},
					File:            "firstfile",
					ExpectedContent: "this is the first file",
				},
				{
					Volume: v1.VolumeSource{
						ConfigMap: &v1.ConfigMapVolumeSource{
							LocalObjectReference: v1.LocalObjectReference{
								Name: config.Prefix + "-map",
							},
							Items: []v1.KeyToPath{
								{
									Key:  "second",
									Path: "secondfile",
								},
							},
						},
					},
					File:            "secondfile",
					ExpectedContent: "this is the second file",
				},
			}
			framework.TestVolumeClient(cs, config, nil, "" /* fsType */, tests)
		})
	})
})
