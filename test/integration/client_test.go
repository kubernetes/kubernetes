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
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
)

func init() {
	requireEtcd()
}

func TestClient(t *testing.T) {
	m := master.New(&master.Config{
		EtcdServers: newEtcdClient().GetCluster(),
	})

	storage, codec := m.API_v1beta1()
	s := httptest.NewServer(apiserver.Handle(storage, codec, "/api/v1beta1/"))

	client := client.NewOrDie(s.URL, nil)

	info, err := client.ServerVersion()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if e, a := version.Get(), *info; !reflect.DeepEqual(e, a) {
		t.Errorf("expected %#v, got %#v", e, a)
	}

	pods, err := client.ListPods(labels.Everything())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(pods.Items) != 0 {
		t.Errorf("expected no pods, got %#v", pods)
	}

	// get a validation error
	pod := api.Pod{
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Version: "v1beta2",
				Containers: []api.Container{
					{
						Name: "test",
					},
				},
			},
		},
	}
	got, err := client.CreatePod(pod)
	if err == nil {
		t.Fatalf("unexpected non-error: %v", err)
	}

	// get a created pod
	pod.DesiredState.Manifest.Containers[0].Image = "an-image"
	got, err = client.CreatePod(pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.ID == "" {
		t.Errorf("unexpected empty pod ID %v", got)
	}

	// pod is shown, but not scheduled
	pods, err = client.ListPods(labels.Everything())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(pods.Items) != 1 {
		t.Errorf("expected one pod, got %#v", pods)
	}
	actual := pods.Items[0]
	if actual.ID != got.ID {
		t.Errorf("expected pod %#v, got %#v", got, actual)
	}
	if actual.CurrentState.Host != "" {
		t.Errorf("expected pod to be unscheduled, got %#v", actual)
	}
}
