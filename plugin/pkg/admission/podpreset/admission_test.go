/*
Copyright 2016 The Kubernetes Authors.

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

package podpreset

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kadmission "k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/settings"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	settingslisters "k8s.io/kubernetes/pkg/client/listers/settings/internalversion"
	"k8s.io/kubernetes/pkg/controller"
)

func TestMergeEnv(t *testing.T) {
	tests := map[string]struct {
		orig       []api.EnvVar
		mod        []api.EnvVar
		result     []api.EnvVar
		shouldFail bool
	}{
		"empty original": {
			mod:        []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "ABC", Value: "value3"}},
			result:     []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "ABC", Value: "value3"}},
			shouldFail: false,
		},
		"good merge": {
			orig:       []api.EnvVar{{Name: "abcd", Value: "value2"}, {Name: "hello", Value: "value3"}},
			mod:        []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "ABC", Value: "value3"}},
			result:     []api.EnvVar{{Name: "abcd", Value: "value2"}, {Name: "hello", Value: "value3"}, {Name: "abc", Value: "value2"}, {Name: "ABC", Value: "value3"}},
			shouldFail: false,
		},
		"conflict": {
			orig:       []api.EnvVar{{Name: "abc", Value: "value3"}},
			mod:        []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "ABC", Value: "value3"}},
			shouldFail: true,
		},
		"one is exact same": {
			orig:       []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "hello", Value: "value3"}},
			mod:        []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "ABC", Value: "value3"}},
			result:     []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "hello", Value: "value3"}, {Name: "ABC", Value: "value3"}},
			shouldFail: false,
		},
	}

	for name, test := range tests {
		result, err := mergeEnv(
			test.orig,
			[]*settings.PodPreset{{Spec: settings.PodPresetSpec{Env: test.mod}}},
		)
		if test.shouldFail && err == nil {
			t.Fatalf("expected test %q to fail but got nil", name)
		}
		if !test.shouldFail && err != nil {
			t.Fatalf("test %q failed: %v", name, err)
		}
		if !reflect.DeepEqual(test.result, result) {
			t.Fatalf("results were not equal for test %q: got %#v; expected: %#v", name, result, test.result)
		}
	}
}

func TestMergeEnvFrom(t *testing.T) {
	tests := map[string]struct {
		orig       []api.EnvFromSource
		mod        []api.EnvFromSource
		result     []api.EnvFromSource
		shouldFail bool
	}{
		"empty original": {
			mod: []api.EnvFromSource{
				{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
				{
					Prefix: "pre_",
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
			},
			result: []api.EnvFromSource{
				{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
				{
					Prefix: "pre_",
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
			},
			shouldFail: false,
		},
		"good merge": {
			orig: []api.EnvFromSource{
				{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "thing"},
					},
				},
			},
			mod: []api.EnvFromSource{
				{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
				{
					Prefix: "pre_",
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
			},
			result: []api.EnvFromSource{
				{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "thing"},
					},
				},
				{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
				{
					Prefix: "pre_",
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
			},
			shouldFail: false,
		},
	}

	for name, test := range tests {
		result, err := mergeEnvFrom(
			test.orig,
			[]*settings.PodPreset{{Spec: settings.PodPresetSpec{EnvFrom: test.mod}}},
		)
		if test.shouldFail && err == nil {
			t.Fatalf("expected test %q to fail but got nil", name)
		}
		if !test.shouldFail && err != nil {
			t.Fatalf("test %q failed: %v", name, err)
		}
		if !reflect.DeepEqual(test.result, result) {
			t.Fatalf("results were not equal for test %q: got %#v; expected: %#v", name, result, test.result)
		}
	}
}

func TestMergeVolumeMounts(t *testing.T) {
	tests := map[string]struct {
		orig       []api.VolumeMount
		mod        []api.VolumeMount
		result     []api.VolumeMount
		shouldFail bool
	}{
		"empty original": {
			mod: []api.VolumeMount{
				{
					Name:      "simply-mounted-volume",
					MountPath: "/opt/",
				},
			},
			result: []api.VolumeMount{
				{
					Name:      "simply-mounted-volume",
					MountPath: "/opt/",
				},
			},
			shouldFail: false,
		},
		"good merge": {
			mod: []api.VolumeMount{
				{
					Name:      "simply-mounted-volume",
					MountPath: "/opt/",
				},
			},
			orig: []api.VolumeMount{
				{
					Name:      "etc-volume",
					MountPath: "/etc/",
				},
			},
			result: []api.VolumeMount{
				{
					Name:      "etc-volume",
					MountPath: "/etc/",
				},
				{
					Name:      "simply-mounted-volume",
					MountPath: "/opt/",
				},
			},
			shouldFail: false,
		},
		"conflict": {
			mod: []api.VolumeMount{
				{
					Name:      "simply-mounted-volume",
					MountPath: "/opt/",
				},
				{
					Name:      "etc-volume",
					MountPath: "/things/",
				},
			},
			orig: []api.VolumeMount{
				{
					Name:      "etc-volume",
					MountPath: "/etc/",
				},
			},
			shouldFail: true,
		},
		"conflict on mount path": {
			mod: []api.VolumeMount{
				{
					Name:      "simply-mounted-volume",
					MountPath: "/opt/",
				},
				{
					Name:      "things-volume",
					MountPath: "/etc/",
				},
			},
			orig: []api.VolumeMount{
				{
					Name:      "etc-volume",
					MountPath: "/etc/",
				},
			},
			shouldFail: true,
		},
		"one is exact same": {
			mod: []api.VolumeMount{
				{
					Name:      "simply-mounted-volume",
					MountPath: "/opt/",
				},
				{
					Name:      "etc-volume",
					MountPath: "/etc/",
				},
			},
			orig: []api.VolumeMount{
				{
					Name:      "etc-volume",
					MountPath: "/etc/",
				},
			},
			result: []api.VolumeMount{
				{
					Name:      "etc-volume",
					MountPath: "/etc/",
				},
				{
					Name:      "simply-mounted-volume",
					MountPath: "/opt/",
				},
			},
			shouldFail: false,
		},
	}

	for name, test := range tests {
		result, err := mergeVolumeMounts(
			test.orig,
			[]*settings.PodPreset{{Spec: settings.PodPresetSpec{VolumeMounts: test.mod}}},
		)
		if test.shouldFail && err == nil {
			t.Fatalf("expected test %q to fail but got nil", name)
		}
		if !test.shouldFail && err != nil {
			t.Fatalf("test %q failed: %v", name, err)
		}
		if !reflect.DeepEqual(test.result, result) {
			t.Fatalf("results were not equal for test %q: got %#v; expected: %#v", name, result, test.result)
		}
	}
}

func TestMergeVolumes(t *testing.T) {
	tests := map[string]struct {
		orig       []api.Volume
		mod        []api.Volume
		result     []api.Volume
		shouldFail bool
	}{
		"empty original": {
			mod: []api.Volume{
				{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				{Name: "vol2", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
			result: []api.Volume{
				{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				{Name: "vol2", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
			shouldFail: false,
		},
		"good merge": {
			orig: []api.Volume{
				{Name: "vol3", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				{Name: "vol4", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
			mod: []api.Volume{
				{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				{Name: "vol2", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
			result: []api.Volume{
				{Name: "vol3", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				{Name: "vol4", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				{Name: "vol2", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
			shouldFail: false,
		},
		"conflict": {
			orig: []api.Volume{
				{Name: "vol3", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				{Name: "vol4", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
			mod: []api.Volume{
				{Name: "vol3", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/etc/apparmor.d"}}},
				{Name: "vol2", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
			shouldFail: true,
		},
		"one is exact same": {
			orig: []api.Volume{
				{Name: "vol3", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				{Name: "vol4", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
			mod: []api.Volume{
				{Name: "vol3", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				{Name: "vol2", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
			result: []api.Volume{
				{Name: "vol3", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				{Name: "vol4", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
				{Name: "vol2", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			},
			shouldFail: false,
		},
	}

	for name, test := range tests {
		result, err := mergeVolumes(
			test.orig,
			[]*settings.PodPreset{{Spec: settings.PodPresetSpec{Volumes: test.mod}}},
		)
		if test.shouldFail && err == nil {
			t.Fatalf("expected test %q to fail but got nil", name)
		}
		if !test.shouldFail && err != nil {
			t.Fatalf("test %q failed: %v", name, err)
		}
		if !reflect.DeepEqual(test.result, result) {
			t.Fatalf("results were not equal for test %q: got %#v; expected: %#v", name, result, test.result)
		}
	}
}

// NewTestAdmission provides an admission plugin with test implementations of internal structs.  It uses
// an authorizer that always returns true.
func NewTestAdmission(lister settingslisters.PodPresetLister, objects ...runtime.Object) kadmission.Interface {
	// Build a test client that the admission plugin can use to look up the service account missing from its cache
	client := fake.NewSimpleClientset(objects...)

	return &podPresetPlugin{
		client:  client,
		Handler: kadmission.NewHandler(kadmission.Create),
		lister:  lister,
	}
}

func TestAdmitConflictWithDifferentNamespaceShouldDoNothing(t *testing.T) {
	containerName := "container"

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mypod",
			Namespace: "namespace",
			Labels: map[string]string{
				"security": "S2",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: containerName,
					Env:  []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "ABC", Value: "value3"}},
				},
			},
		},
	}

	pip := &settings.PodPreset{
		ObjectMeta: v1.ObjectMeta{
			Name:      "hello",
			Namespace: "othernamespace",
		},
		Spec: settings.PodPresetSpec{
			Selector: v1.LabelSelector{
				MatchExpressions: []v1.LabelSelectorRequirement{
					{
						Key:      "security",
						Operator: v1.LabelSelectorOpIn,
						Values:   []string{"S2"},
					},
				},
			},
			Env: []api.EnvVar{{Name: "abc", Value: "value"}, {Name: "ABC", Value: "value"}},
		},
	}

	err := admitPod(pod, pip)
	if err != nil {
		t.Fatal(err)
	}
}

func TestAdmitConflictWithNonMatchingLabelsShouldNotError(t *testing.T) {
	containerName := "container"

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mypod",
			Namespace: "namespace",
			Labels: map[string]string{
				"security": "S2",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: containerName,
					Env:  []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "ABC", Value: "value3"}},
				},
			},
		},
	}

	pip := &settings.PodPreset{
		ObjectMeta: v1.ObjectMeta{
			Name:      "hello",
			Namespace: "namespace",
		},
		Spec: settings.PodPresetSpec{
			Selector: v1.LabelSelector{
				MatchExpressions: []v1.LabelSelectorRequirement{
					{
						Key:      "security",
						Operator: v1.LabelSelectorOpIn,
						Values:   []string{"S3"},
					},
				},
			},
			Env: []api.EnvVar{{Name: "abc", Value: "value"}, {Name: "ABC", Value: "value"}},
		},
	}

	err := admitPod(pod, pip)
	if err != nil {
		t.Fatal(err)
	}
}

func TestAdmitConflictShouldNotModifyPod(t *testing.T) {
	containerName := "container"

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mypod",
			Namespace: "namespace",
			Labels: map[string]string{
				"security": "S2",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: containerName,
					Env:  []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "ABC", Value: "value3"}},
				},
			},
		},
	}
	origPod := *pod

	pip := &settings.PodPreset{
		ObjectMeta: v1.ObjectMeta{
			Name:      "hello",
			Namespace: "namespace",
		},
		Spec: settings.PodPresetSpec{
			Selector: v1.LabelSelector{
				MatchExpressions: []v1.LabelSelectorRequirement{
					{
						Key:      "security",
						Operator: v1.LabelSelectorOpIn,
						Values:   []string{"S2"},
					},
				},
			},
			Env: []api.EnvVar{{Name: "abc", Value: "value"}, {Name: "ABC", Value: "value"}},
		},
	}

	err := admitPod(pod, pip)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(&origPod, pod) {
		t.Fatalf("pod should not get modified in case of conflict origPod: %+v got: %+v", &origPod, pod)
	}
}

func TestAdmit(t *testing.T) {
	containerName := "container"

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mypod",
			Namespace: "namespace",
			Labels: map[string]string{
				"security": "S2",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: containerName,
					Env:  []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "ABCD", Value: "value3"}},
				},
			},
		},
	}

	pip := &settings.PodPreset{
		ObjectMeta: v1.ObjectMeta{
			Name:      "hello",
			Namespace: "namespace",
		},
		Spec: settings.PodPresetSpec{
			Selector: v1.LabelSelector{
				MatchExpressions: []v1.LabelSelectorRequirement{
					{
						Key:      "security",
						Operator: v1.LabelSelectorOpIn,
						Values:   []string{"S2"},
					},
				},
			},
			Volumes: []api.Volume{{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}},
			Env:     []api.EnvVar{{Name: "abcd", Value: "value"}, {Name: "ABC", Value: "value"}},
			EnvFrom: []api.EnvFromSource{
				{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
				{
					Prefix: "pre_",
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
			},
		},
	}

	err := admitPod(pod, pip)
	if err != nil {
		t.Fatal(err)
	}
}

func TestAdmitMirrorPod(t *testing.T) {
	containerName := "container"

	mirrorPod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mypod",
			Namespace: "namespace",
			Labels: map[string]string{
				"security": "S2",
			},
			Annotations: map[string]string{api.MirrorPodAnnotationKey: "mirror"},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: containerName,
				},
			},
		},
	}

	pip := &settings.PodPreset{
		ObjectMeta: v1.ObjectMeta{
			Name:      "hello",
			Namespace: "namespace",
		},
		Spec: settings.PodPresetSpec{
			Selector: v1.LabelSelector{
				MatchExpressions: []v1.LabelSelectorRequirement{
					{
						Key:      "security",
						Operator: v1.LabelSelectorOpIn,
						Values:   []string{"S2"},
					},
				},
			},
			Volumes: []api.Volume{{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}},
			Env:     []api.EnvVar{{Name: "abcd", Value: "value"}, {Name: "ABC", Value: "value"}},
			EnvFrom: []api.EnvFromSource{
				{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
				{
					Prefix: "pre_",
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
			},
		},
	}

	if err := admitPod(mirrorPod, pip); err != nil {
		t.Fatal(err)
	}

	container := mirrorPod.Spec.Containers[0]
	if len(mirrorPod.Spec.Volumes) != 0 ||
		len(container.VolumeMounts) != 0 ||
		len(container.Env) != 0 ||
		len(container.EnvFrom) != 0 {
		t.Fatalf("mirror pod is updated by PodPreset admission:\n\tVolumes got %d, expected 0\n\tVolumeMounts go %d, expected 0\n\tEnv got, %d expected 0\n\tEnvFrom got %d, expected 0", len(mirrorPod.Spec.Volumes), len(container.VolumeMounts), len(container.Env), len(container.EnvFrom))
	}
}

func TestExclusionNoAdmit(t *testing.T) {
	containerName := "container"

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mypod",
			Namespace: "namespace",
			Labels: map[string]string{
				"security": "S2",
			},
			Annotations: map[string]string{
				api.PodPresetOptOutAnnotationKey: "true",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: containerName,
					Env:  []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "ABCD", Value: "value3"}},
				},
			},
		},
	}

	pip := &settings.PodPreset{
		ObjectMeta: v1.ObjectMeta{
			Name:      "hello",
			Namespace: "namespace",
		},
		Spec: settings.PodPresetSpec{
			Selector: v1.LabelSelector{
				MatchExpressions: []v1.LabelSelectorRequirement{
					{
						Key:      "security",
						Operator: v1.LabelSelectorOpIn,
						Values:   []string{"S2"},
					},
				},
			},
			Volumes: []api.Volume{{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}},
			Env:     []api.EnvVar{{Name: "abcd", Value: "value"}, {Name: "ABC", Value: "value"}},
			EnvFrom: []api.EnvFromSource{
				{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
				{
					Prefix: "pre_",
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
			},
		},
	}
	originalPod, err := api.Scheme.Copy(pod)
	if err != nil {
		t.Fatal(err)
	}

	err = admitPod(pod, pip)
	if err != nil {
		t.Fatal(err)
	}

	// verify PodSpec has not been mutated
	if !reflect.DeepEqual(pod, originalPod) {
		t.Fatalf("Expected pod spec of '%v' to be unchanged", pod.Name)
	}
}

func TestAdmitEmptyPodNamespace(t *testing.T) {
	containerName := "container"

	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "mypod",
			Labels: map[string]string{
				"security": "S2",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name: containerName,
					Env:  []api.EnvVar{{Name: "abc", Value: "value2"}, {Name: "ABCD", Value: "value3"}},
				},
			},
		},
	}

	pip := &settings.PodPreset{
		ObjectMeta: v1.ObjectMeta{
			Name:      "hello",
			Namespace: "different", // (pod will be submitted to namespace 'namespace')
		},
		Spec: settings.PodPresetSpec{
			Selector: v1.LabelSelector{
				MatchExpressions: []v1.LabelSelectorRequirement{
					{
						Key:      "security",
						Operator: v1.LabelSelectorOpIn,
						Values:   []string{"S2"},
					},
				},
			},
			Volumes: []api.Volume{{Name: "vol", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}}},
			Env:     []api.EnvVar{{Name: "abcd", Value: "value"}, {Name: "ABC", Value: "value"}},
			EnvFrom: []api.EnvFromSource{
				{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
				{
					Prefix: "pre_",
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{Name: "abc"},
					},
				},
			},
		},
	}
	originalPod, err := api.Scheme.Copy(pod)
	if err != nil {
		t.Fatal(err)
	}

	err = admitPod(pod, pip)
	if err != nil {
		t.Fatal(err)
	}

	// verify PodSpec has not been mutated
	if !reflect.DeepEqual(pod, originalPod) {
		t.Fatalf("pod should not get modified in case of emptyNamespace origPod: %+v got: %+v", originalPod, pod)
	}
}

func admitPod(pod *api.Pod, pip *settings.PodPreset) error {
	informerFactory := informers.NewSharedInformerFactory(nil, controller.NoResyncPeriodFunc())
	store := informerFactory.Settings().InternalVersion().PodPresets().Informer().GetStore()
	store.Add(pip)
	plugin := NewTestAdmission(informerFactory.Settings().InternalVersion().PodPresets().Lister())
	attrs := kadmission.NewAttributesRecord(
		pod,
		nil,
		api.Kind("Pod").WithVersion("version"),
		"namespace",
		"",
		api.Resource("pods").WithVersion("version"),
		"",
		kadmission.Create,
		&user.DefaultInfo{},
	)

	err := plugin.Admit(attrs)
	if err != nil {
		return err
	}

	return nil
}
