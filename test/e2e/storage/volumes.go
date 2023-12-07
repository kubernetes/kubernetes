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
	"context"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

// These tests need privileged containers, which are disabled by default.
var _ = utils.SIGDescribe("Volumes", func() {
	f := framework.NewDefaultFramework("volume")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	// note that namespace deletion is handled by delete-namespace flag
	// filled inside BeforeEach
	var cs clientset.Interface
	var namespace *v1.Namespace

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		namespace = f.Namespace
	})

	ginkgo.Describe("ConfigMap", func() {
		ginkgo.It("should be mountable", func(ctx context.Context) {
			config := e2evolume.TestConfig{
				Namespace: namespace.Name,
				Prefix:    "configmap",
			}
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
			if _, err := cs.CoreV1().ConfigMaps(namespace.Name).Create(ctx, configMap, metav1.CreateOptions{}); err != nil {
				framework.Failf("unable to create test configmap: %v", err)
			}
			defer func() {
				_ = cs.CoreV1().ConfigMaps(namespace.Name).Delete(ctx, configMap.Name, metav1.DeleteOptions{})
			}()

			// Test one ConfigMap mounted several times to test #28502
			tests := []e2evolume.Test{
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
			e2evolume.TestVolumeClient(ctx, f, config, nil, "" /* fsType */, tests)
		})
	})
})
