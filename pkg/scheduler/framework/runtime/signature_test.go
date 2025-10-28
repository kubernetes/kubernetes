/*
Copyright 2025 The Kubernetes Authors.

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

package runtime

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestNewPodSignatureMaker(t *testing.T) {
	s := newPodSignatureMaker()
	if s == nil {
		t.Fatal("newPodSignatureMaker() returned nil")
	}
	if !s.signable {
		t.Errorf("newPodSignatureMaker() signable = %v, want true", s.signable)
	}
	if len(s.elements.Pod) != 0 {
		t.Errorf("newPodSignatureMaker() pod elements map should be empty, got %v", s.elements.Pod)
	}
	if len(s.elements.Plugin) != 0 {
		t.Errorf("newPodSignatureMaker() plugin elements map should be empty, got %v", s.elements.Plugin)
	}
}

func TestUnsignable(t *testing.T) {
	s := newPodSignatureMaker()
	s.Unsignable()
	if s.signable {
		t.Errorf("Unsignable() did not set signable to false")
	}
}

func TestAddPodElement(t *testing.T) {
	tests := []struct {
		name        string
		initialPod  map[string]string
		podPath     string
		object      any
		wantErr     bool
		expectedPod map[string]string
	}{
		{
			name:    "add_new_string",
			podPath: "spec.nodename",
			object:  "test-node",
			wantErr: false,
			expectedPod: map[string]string{
				"spec.nodename": "\"test-node\"",
			},
		},
		{
			name:    "add_new_struct",
			podPath: "spec.containers",
			object:  []v1.Container{{Name: "c1"}},
			wantErr: false,
			expectedPod: map[string]string{
				"spec.containers": "[{\"name\":\"c1\",\"resources\":{}}]",
			},
		},
		{
			name: "add_duplicate",
			initialPod: map[string]string{
				"spec.nodename": "\"initial-node\"",
			},
			podPath: "spec.nodename",
			object:  "new-node",
			wantErr: false,
			expectedPod: map[string]string{
				"spec.nodename": "\"initial-node\"", // Should not change
			},
		},
		{
			name:        "json_marshal_error",
			podPath:     "badelement",
			object:      make(chan int), // Channels cannot be marshalled to JSON
			wantErr:     true,
			expectedPod: map[string]string{
				// No changes expected
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := newPodSignatureMaker()
			if tc.initialPod != nil {
				s.elements.Pod = tc.initialPod
			}

			err := s.AddPodElement(tc.podPath, tc.object)

			if (err != nil) != tc.wantErr {
				t.Errorf("AddPodElement(%q, %v) error = %v, wantErr %v", tc.podPath, tc.object, err, tc.wantErr)
			}

			if diff := cmp.Diff(tc.expectedPod, s.elements.Pod); diff != "" {
				t.Errorf("AddPodElement(%q, %v) pod elements map diff (-want +got):\n%s", tc.podPath, tc.object, diff)
			}
		})
	}
}

func TestAddPluginElement(t *testing.T) {
	s := newPodSignatureMaker()
	pluginName := "myplugin"
	pluginData := map[string]string{"key": "value"}
	expectedJSON := "{\"key\":\"value\"}"

	err := s.AddPluginElement(pluginName, pluginData)
	if err != nil {
		t.Fatalf("AddPluginElement(%q, %v) failed: %v", pluginName, pluginData, err)
	}

	if val, ok := s.elements.Plugin[pluginName]; !ok || val != expectedJSON {
		t.Errorf("AddPluginElement(%q, %v) plugin map got %v, want %v", pluginName, pluginData, s.elements.Plugin, map[string]string{pluginName: expectedJSON})
	}

	// Test error case
	err = s.AddPluginElement("badplugin", make(chan int))
	if err == nil {
		t.Errorf("AddPluginElement with unmarshallable object did not return an error")
	}
}

func TestMarshal(t *testing.T) {
	s := newPodSignatureMaker()
	if err := s.AddPodElement("spec.nodename", "node1"); err != nil {
		t.Fatalf("Add failed %v", err)
	}
	if err := s.AddPluginElement("myplugin", "data1"); err != nil {
		t.Fatalf("Add failed %v", err)
	}

	expectedObj := Elements{
		Pod:    map[string]string{"spec.nodename": "\"node1\""},
		Plugin: map[string]string{"myplugin": "\"data1\""},
	}

	gotJSON, err := s.Marshal()
	if err != nil {
		t.Fatalf("Marshal() failed: %v", err)
	}

	// Unmarshal to compare maps because key order is not guaranteed in marshalled JSON
	var gotObj Elements
	if err := json.Unmarshal(gotJSON, &gotObj); err != nil {
		t.Fatalf("Failed to unmarshal result: %v", err)
	}

	if diff := cmp.Diff(expectedObj, gotObj); diff != "" {
		t.Errorf("Marshal() diff (-want +got):\n%s", diff)
	}
}

func TestAddNonPluginElements(t *testing.T) {
	pod := &v1.Pod{
		Spec: v1.PodSpec{
			SchedulerName: "my-scheduler",
		},
	}
	s := newPodSignatureMaker()
	s.AddNonPluginElements(pod)

	expectedPodMap := map[string]string{
		"Spec.SchedulerName": "\"my-scheduler\"",
	}

	if diff := cmp.Diff(expectedPodMap, s.elements.Pod); diff != "" {
		t.Errorf("AddNonPluginElements() pod elements map diff (-want +got):\n%s", diff)
	}
}

func TestAddSignatureVolumes(t *testing.T) {
	volEmptyDir := v1.Volume{Name: "empty", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}
	volHostPath := v1.Volume{Name: "host", VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{Path: "/tmp"}}}
	volConfigMap := v1.Volume{Name: "cm", VolumeSource: v1.VolumeSource{ConfigMap: &v1.ConfigMapVolumeSource{}}}
	volSecret := v1.Volume{Name: "secret", VolumeSource: v1.VolumeSource{Secret: &v1.SecretVolumeSource{}}}

	tests := []struct {
		name          string
		podVolumes    []v1.Volume
		expectedVols  []v1.Volume
		expectAdded   bool
		initialCaches map[string]string
	}{
		{
			name:         "no_volumes",
			podVolumes:   []v1.Volume{},
			expectedVols: []v1.Volume{},
			expectAdded:  true,
		},
		{
			name:         "only_allowed_volumes",
			podVolumes:   []v1.Volume{volEmptyDir, volHostPath},
			expectedVols: []v1.Volume{volEmptyDir, volHostPath},
			expectAdded:  true,
		},
		{
			name:         "mixed_volumes",
			podVolumes:   []v1.Volume{volEmptyDir, volConfigMap, volHostPath, volSecret},
			expectedVols: []v1.Volume{volEmptyDir, volHostPath},
			expectAdded:  true,
		},
		{
			name:         "only_excluded_volumes",
			podVolumes:   []v1.Volume{volConfigMap, volSecret},
			expectedVols: []v1.Volume{},
			expectAdded:  true,
		},
		{
			name:       "volumes_already_exist",
			podVolumes: []v1.Volume{volEmptyDir},
			initialCaches: map[string]string{
				"_SignatureVolumes": "[]",
			},
			expectAdded: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "test-pod"},
				Spec: v1.PodSpec{
					Volumes: tc.podVolumes,
				},
			}
			s := newPodSignatureMaker()
			if tc.initialCaches != nil {
				s.elements.Pod = tc.initialCaches
			}

			err := s.AddSignatureVolumes(pod)
			if err != nil {
				t.Fatalf("AddSignatureVolumes() failed: %v", err)
			}

			volJSON, found := s.elements.Pod["_SignatureVolumes"]
			if !tc.expectAdded {
				if found && volJSON != tc.initialCaches["_SignatureVolumes"] {
					t.Errorf("AddSignatureVolumes should not have modified existing _SignatureVolumes")
				}
				return
			}

			if !found {
				t.Fatalf("AddSignatureVolumes() did not add _SignatureVolumes to pod elements")
			}

			var gotVols []v1.Volume
			if err := json.Unmarshal([]byte(volJSON), &gotVols); err != nil {
				t.Fatalf("Failed to unmarshal _SignatureVolumes: %v", err)
			}

			if diff := cmp.Diff(tc.expectedVols, gotVols); diff != "" {
				t.Errorf("AddSignatureVolumes() volumes diff (-want +got):\n%s", diff)
			}
		})
	}
}
