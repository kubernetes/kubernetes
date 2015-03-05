// +build integration,!no-etcd

/*
Copyright 2014 Google Inc. All rights reserved.

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

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/admission/admit"
)

func init() {
	requireEtcd()
}

func TestClient(t *testing.T) {
	helper, err := master.NewEtcdHelper(newEtcdClient(), "v1beta1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	m = master.New(&master.Config{
		EtcdHelper:        helper,
		KubeletClient:     client.FakeKubeletClient{},
		EnableLogsSupport: false,
		EnableProfiling:   true,
		EnableUISupport:   false,
		EnableV1Beta3:     true,
		APIPrefix:         "/api",
		Authorizer:        apiserver.NewAlwaysAllowAuthorizer(),
		AdmissionControl:  admit.NewAlwaysAdmit(),
	})

	testCases := []string{
		"v1beta1",
		"v1beta2",
		"v1beta3",
	}
	for _, apiVersion := range testCases {
		ns := api.NamespaceDefault
		deleteAllEtcdKeys()
		client := client.NewOrDie(&client.Config{Host: s.URL, Version: apiVersion})

		info, err := client.ServerVersion()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if e, a := version.Get(), *info; !reflect.DeepEqual(e, a) {
			t.Errorf("expected %#v, got %#v", e, a)
		}

		pods, err := client.Pods(ns).List(labels.Everything())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(pods.Items) != 0 {
			t.Errorf("expected no pods, got %#v", pods)
		}

		// get a validation error
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				GenerateName: "test",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test",
					},
				},
			},
		}

		got, err := client.Pods(ns).Create(pod)
		if err == nil {
			t.Fatalf("unexpected non-error: %v", got)
		}

		// get a created pod
		pod.Spec.Containers[0].Image = "an-image"
		got, err = client.Pods(ns).Create(pod)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.Name == "" {
			t.Errorf("unexpected empty pod Name %v", got)
		}

		// pod is shown, but not scheduled
		pods, err = client.Pods(ns).List(labels.Everything())
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(pods.Items) != 1 {
			t.Errorf("expected one pod, got %#v", pods)
		}
		actual := pods.Items[0]
		if actual.Name != got.Name {
			t.Errorf("expected pod %#v, got %#v", got, actual)
		}
		if actual.Status.Host != "" {
			t.Errorf("expected pod to be unscheduled, got %#v", actual)
		}
	}
}
