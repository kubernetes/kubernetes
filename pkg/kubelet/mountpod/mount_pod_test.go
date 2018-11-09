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

package mountpod

import (
	"io/ioutil"
	"os"
	"path"
	"testing"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/kubelet/configmap"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	podtest "k8s.io/kubernetes/pkg/kubelet/pod/testing"
	"k8s.io/kubernetes/pkg/kubelet/secret"
)

func TestGetVolumeExec(t *testing.T) {
	// prepare PodManager
	pods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "bar",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "baz"},
				},
			},
		},
	}
	fakeSecretManager := secret.NewFakeManager()
	fakeConfigMapManager := configmap.NewFakeManager()
	podManager := kubepod.NewBasicPodManager(
		podtest.NewFakeMirrorClient(), fakeSecretManager, fakeConfigMapManager, podtest.NewMockCheckpointManager())
	podManager.SetPods(pods)

	// Prepare fake /var/lib/kubelet
	basePath, err := utiltesting.MkTmpdir("kubelet")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(basePath)
	regPath := path.Join(basePath, "plugin-containers")

	mgr, err := NewManager(basePath, podManager)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name        string
		json        string
		expectError bool
	}{
		{
			"invalid json",
			"{{{}",
			true,
		},
		{
			"missing json",
			"", // this means no json file should be created
			false,
		},
		{
			"missing podNamespace",
			`{"podName": "foo", "podUID": "87654321", "containerName": "baz"}`,
			true,
		},
		{
			"missing podName",
			`{"podNamespace": "bar", "podUID": "87654321", "containerName": "baz"}`,
			true,
		},
		{
			"missing containerName",
			`{"podNamespace": "bar", "podName": "foo", "podUID": "87654321"}`,
			true,
		},
		{
			"missing podUID",
			`{"podNamespace": "bar", "podName": "foo", "containerName": "baz"}`,
			true,
		},
		{
			"missing pod",
			`{"podNamespace": "bar", "podName": "non-existing-pod", "podUID": "12345678", "containerName": "baz"}`,
			true,
		},
		{
			"invalid uid",
			`{"podNamespace": "bar", "podName": "foo", "podUID": "87654321", "containerName": "baz"}`,
			true,
		},
		{
			"invalid container",
			`{"podNamespace": "bar", "podName": "foo", "podUID": "12345678", "containerName": "invalid"}`,
			true,
		},
		{
			"valid pod",
			`{"podNamespace": "bar", "podName": "foo", "podUID": "12345678", "containerName": "baz"}`,
			false,
		},
	}
	for _, test := range tests {
		p := path.Join(regPath, "kubernetes.io~glusterfs.json")
		if len(test.json) > 0 {
			if err := ioutil.WriteFile(p, []byte(test.json), 0600); err != nil {
				t.Errorf("test %q: error writing %s: %v", test.name, p, err)
				continue
			}
		} else {
			// "" means no JSON file
			os.Remove(p)
		}
		pod, container, err := mgr.GetMountPod("kubernetes.io/glusterfs")
		if err != nil {
			klog.V(5).Infof("test %q returned error %s", test.name, err)
		}
		if err == nil && test.expectError {
			t.Errorf("test %q: expected error, got none", test.name)
		}
		if err != nil && !test.expectError {
			t.Errorf("test %q: unexpected error: %v", test.name, err)
		}

		if err == nil {
			// Pod must be returned when the json file was not empty
			if pod == nil && len(test.json) != 0 {
				t.Errorf("test %q: expected exec, got nil", test.name)
			}
			// Both pod and container must be returned
			if pod != nil && len(container) == 0 {
				t.Errorf("test %q: expected container name, got %q", test.name, container)
			}
		}
	}
}
