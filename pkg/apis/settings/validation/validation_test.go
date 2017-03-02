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

package validation

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/settings"
)

func TestValidateEmptyPodPreset(t *testing.T) {
	emptyPodPreset := &settings.PodPreset{
		Spec: settings.PodPresetSpec{},
	}

	errList := ValidatePodPreset(emptyPodPreset)
	if errList == nil {
		t.Fatal("empty pod preset should return an error")
	}
}

func TestValidateEmptyPodPresetItems(t *testing.T) {
	emptyPodPreset := &settings.PodPreset{
		ObjectMeta: v1.ObjectMeta{
			Name:      "hello",
			Namespace: "sample",
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
		},
	}

	errList := ValidatePodPreset(emptyPodPreset)
	if !strings.Contains(errList.ToAggregate().Error(), "must specify at least one") {
		t.Fatal("empty pod preset with label selector should return an error")
	}
}

func TestValidatePodPresets(t *testing.T) {
	p := &settings.PodPreset{
		ObjectMeta: v1.ObjectMeta{
			Name:      "hello",
			Namespace: "sample",
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
			Env:     []api.EnvVar{{Name: "abc", Value: "value"}, {Name: "ABC", Value: "value"}},
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

	errList := ValidatePodPreset(p)
	if errList != nil {
		if errList.ToAggregate() != nil {
			t.Fatalf("errors: %#v", errList.ToAggregate().Error())
		}
	}

	p = &settings.PodPreset{
		ObjectMeta: v1.ObjectMeta{
			Name:      "hello",
			Namespace: "sample",
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
			Env:     []api.EnvVar{{Name: "abc", Value: "value"}, {Name: "ABC", Value: "value"}},
			VolumeMounts: []api.VolumeMount{
				{Name: "vol", MountPath: "/foo"},
			},
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

	errList = ValidatePodPreset(p)
	if errList != nil {
		if errList.ToAggregate() != nil {
			t.Fatalf("errors: %#v", errList.ToAggregate().Error())
		}
	}
}

func TestValidatePodPresetsiVolumeMountError(t *testing.T) {
	p := &settings.PodPreset{
		ObjectMeta: v1.ObjectMeta{
			Name:      "hello",
			Namespace: "sample",
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
			VolumeMounts: []api.VolumeMount{
				{Name: "dne", MountPath: "/foo"},
			},
			Env: []api.EnvVar{{Name: "abc", Value: "value"}, {Name: "ABC", Value: "value"}},
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

	errList := ValidatePodPreset(p)
	if !strings.Contains(errList.ToAggregate().Error(), "spec.volumeMounts[0].name: Not found") {
		t.Fatal("should have returned error for volume that does not exist")
	}
}
