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

package clusterinfo

import (
	"io/ioutil"
	"os"
	"testing"
	"text/template"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
)

var testConfigTempl = template.Must(template.New("test").Parse(`apiVersion: v1
clusters:
- cluster:
    server: {{.Server}}
  name: kubernetes
contexts:
- context:
    cluster: kubernetes
    user: kubernetes-admin
  name: kubernetes-admin@kubernetes
current-context: kubernetes-admin@kubernetes
kind: Config
preferences: {}
users:
- name: kubernetes-admin`))

func TestCreateBootstrapConfigMapIfNotExists(t *testing.T) {
	tests := []struct {
		name      string
		createErr error
		updateErr error
		expectErr bool
	}{
		{
			"successful case should have no error",
			nil,
			nil,
			false,
		},
		{
			"if both create and update errors, return error",
			apierrors.NewAlreadyExists(api.Resource("configmaps"), "test"),
			apierrors.NewUnauthorized("go away!"),
			true,
		},
		{
			"unexpected error should be returned",
			apierrors.NewUnauthorized("go away!"),
			nil,
			true,
		},
	}

	servers := []struct {
		Server string
	}{
		{Server: "https://10.128.0.6:6443"},
		{Server: "https://[2001:db8::6]:3446"},
	}

	for _, server := range servers {
		file, err := ioutil.TempFile("", "")
		if err != nil {
			t.Fatalf("could not create tempfile: %v", err)
		}
		defer os.Remove(file.Name())

		if err := testConfigTempl.Execute(file, server); err != nil {
			t.Fatalf("could not write to tempfile: %v", err)
		}

		if err := file.Close(); err != nil {
			t.Fatalf("could not close tempfile: %v", err)
		}

		for _, tc := range tests {
			client := clientsetfake.NewSimpleClientset()
			if tc.createErr != nil {
				client.PrependReactor("create", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
					return true, nil, tc.createErr
				})
			}

			err := CreateBootstrapConfigMapIfNotExists(client, file.Name())
			if tc.expectErr && err == nil {
				t.Errorf("CreateBootstrapConfigMapIfNotExists(%s) wanted error, got nil", tc.name)
			} else if !tc.expectErr && err != nil {
				t.Errorf("CreateBootstrapConfigMapIfNotExists(%s) returned unexpected error: %v", tc.name, err)
			}
		}
	}
}
