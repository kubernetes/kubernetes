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

// This file tests use of the secrets API resource.

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/admission/admit"
)

func init() {
	requireEtcd()
}

func deletePodOrErrorf(t *testing.T, c *client.Client, ns, name string) {
	if err := c.Pods(ns).Delete(name, nil); err != nil {
		t.Errorf("unable to delete pod %v: %v", name, err)
	}
}
func deleteSecretOrErrorf(t *testing.T, c *client.Client, ns, name string) {
	if err := c.Secrets(ns).Delete(name); err != nil {
		t.Errorf("unable to delete secret %v: %v", name, err)
	}
}

// TestSecrets tests apiserver-side behavior of creation of secret objects and their use by pods.
func TestSecrets(t *testing.T) {
	helper, err := master.NewEtcdHelper(newEtcdClient(), testapi.Version(), etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		EtcdHelper:            helper,
		KubeletClient:         client.FakeKubeletClient{},
		EnableCoreControllers: true,
		EnableLogsSupport:     false,
		EnableUISupport:       false,
		EnableIndex:           true,
		APIPrefix:             "/api",
		Authorizer:            apiserver.NewAlwaysAllowAuthorizer(),
		AdmissionControl:      admit.NewAlwaysAdmit(),
	})

	deleteAllEtcdKeys()
	client := client.NewOrDie(&client.Config{Host: s.URL, Version: testapi.Version()})
	DoTestSecrets(t, client, testapi.Version())
}

// DoTestSecrets test secrets for one api version.
func DoTestSecrets(t *testing.T, client *client.Client, apiVersion string) {
	// Make a secret object.
	ns := "ns"
	s := api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name:      "secret",
			Namespace: ns,
		},
		Data: map[string][]byte{
			"data": []byte("value1\n"),
		},
	}

	if _, err := client.Secrets(s.Namespace).Create(&s); err != nil {
		t.Errorf("unable to create test secret: %v", err)
	}
	defer deleteSecretOrErrorf(t, client, s.Namespace, s.Name)

	// Template for pods that use a secret.
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "XXX",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "secvol",
					VolumeSource: api.VolumeSource{
						Secret: &api.SecretVolumeSource{
							SecretName: "secret",
						},
					},
				},
			},
			Containers: []api.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "secvol",
							MountPath: "/fake/path",
							ReadOnly:  true,
						},
					},
				},
			},
		},
	}

	// Create a pod to consume secret.
	pod.ObjectMeta.Name = "uses-secret"
	if _, err := client.Pods(ns).Create(pod); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	defer deletePodOrErrorf(t, client, ns, pod.Name)

	// Create a pod that consumes non-existent secret.
	pod.ObjectMeta.Name = "uses-non-existent-secret"
	if _, err := client.Pods(ns).Create(pod); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}
	defer deletePodOrErrorf(t, client, ns, pod.Name)
	// This pod may fail to run, but we don't currently prevent this, and this
	// test can't check whether the kubelet actually pulls the secret.

	// Verifying contents of the volumes is out of scope for a
	// apiserver<->kubelet integration test.  It is covered by an e2e test.
}
