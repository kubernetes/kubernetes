// +build integration,!no-etcd

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package integration

// This file tests use of the configMap API resource.

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestConfigMap tests apiserver-side behavior of creation of ConfigMaps and pods that consume them.
func TestConfigMap(t *testing.T) {
	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("Error in bringing up the master: %v", err)
	}

	framework.DeleteAllEtcdKeys()
	client := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	DoTestConfigMap(t, client)
}

func DoTestConfigMap(t *testing.T, client *client.Client) {
	ns := "ns"
	cfg := api.ConfigMap{
		ObjectMeta: api.ObjectMeta{
			Name:      "configmap",
			Namespace: ns,
		},
		Data: map[string]string{
			"data-1": "value-1",
			"data-2": "value-2",
			"data-3": "value-3",
		},
	}

	if _, err := client.ConfigMaps(cfg.Namespace).Create(&cfg); err != nil {
		t.Errorf("unable to create test configMap: %v", err)
	}
	defer deleteConfigMapOrErrorf(t, client, cfg.Namespace, cfg.Name)

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "XXX",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
					Env: []api.EnvVar{
						{
							Name: "CONFIG_DATA_1",
							ValueFrom: &api.EnvVarSource{
								ConfigMapKeyRef: &api.ConfigMapKeySelector{
									LocalObjectReference: api.LocalObjectReference{
										Name: "configmap",
									},
									Key: "data-1",
								},
							},
						},
						{
							Name: "CONFIG_DATA_2",
							ValueFrom: &api.EnvVarSource{
								ConfigMapKeyRef: &api.ConfigMapKeySelector{
									LocalObjectReference: api.LocalObjectReference{
										Name: "configmap",
									},
									Key: "data-2",
								},
							},
						}, {
							Name: "CONFIG_DATA_3",
							ValueFrom: &api.EnvVarSource{
								ConfigMapKeyRef: &api.ConfigMapKeySelector{
									LocalObjectReference: api.LocalObjectReference{
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
	if _, err := client.Pods(ns).Create(pod); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	defer deletePodOrErrorf(t, client, ns, pod.Name)
}

func deleteConfigMapOrErrorf(t *testing.T, c *client.Client, ns, name string) {
	if err := c.ConfigMaps(ns).Delete(name); err != nil {
		t.Errorf("unable to delete ConfigMap %v: %v", name, err)
	}
}
