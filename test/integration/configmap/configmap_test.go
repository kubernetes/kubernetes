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

package configmap

// This file tests use of the configMap API resource.

import (
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestConfigMap tests apiserver-side behavior of creation of ConfigMaps and pods that consume them.
func TestConfigMap(t *testing.T) {
	_, s, closeFn := framework.RunAMaster(nil)
	defer closeFn()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})

	ns := framework.CreateTestingNamespace("config-map", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	DoTestConfigMap(t, client, ns)
}

func DoTestConfigMap(t *testing.T, client clientset.Interface, ns *v1.Namespace) {
	cfg := v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "configmap",
			Namespace: ns.Name,
		},
		Data: map[string]string{
			"data-1": "value-1",
			"data-2": "value-2",
			"data-3": "value-3",
		},
	}

	if _, err := client.CoreV1().ConfigMaps(cfg.Namespace).Create(&cfg); err != nil {
		t.Errorf("unable to create test configMap: %v", err)
	}
	defer deleteConfigMapOrErrorf(t, client, cfg.Namespace, cfg.Name)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "XXX",
			Namespace: ns.Name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
					Env: []v1.EnvVar{
						{
							Name: "CONFIG_DATA_1",
							ValueFrom: &v1.EnvVarSource{
								ConfigMapKeyRef: &v1.ConfigMapKeySelector{
									LocalObjectReference: v1.LocalObjectReference{
										Name: "configmap",
									},
									Key: "data-1",
								},
							},
						},
						{
							Name: "CONFIG_DATA_2",
							ValueFrom: &v1.EnvVarSource{
								ConfigMapKeyRef: &v1.ConfigMapKeySelector{
									LocalObjectReference: v1.LocalObjectReference{
										Name: "configmap",
									},
									Key: "data-2",
								},
							},
						}, {
							Name: "CONFIG_DATA_3",
							ValueFrom: &v1.EnvVarSource{
								ConfigMapKeyRef: &v1.ConfigMapKeySelector{
									LocalObjectReference: v1.LocalObjectReference{
										Name: "configmap",
									},
									Key: "data-3",
								},
							},
						},
					},
				},
			},
		},
	}

	pod.ObjectMeta.Name = "uses-configmap"
	if _, err := client.CoreV1().Pods(ns.Name).Create(pod); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	defer integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
}

func deleteConfigMapOrErrorf(t *testing.T, c clientset.Interface, ns, name string) {
	if err := c.CoreV1().ConfigMaps(ns).Delete(name, nil); err != nil {
		t.Errorf("unable to delete ConfigMap %v: %v", name, err)
	}
}
