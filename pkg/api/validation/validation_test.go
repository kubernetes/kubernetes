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

package validation

import (
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func expectPrefix(t *testing.T, prefix string, errs errors.ValidationErrorList) {
	for i := range errs {
		if f, p := errs[i].(errors.ValidationError).Field, prefix; !strings.HasPrefix(f, p) {
			t.Errorf("expected prefix '%s' for field '%s' (%v)", p, f, errs[i])
		}
	}
}

func TestValidateVolumes(t *testing.T) {
	successCase := []api.Volume{
		{Name: "abc"},
		{Name: "123", Source: &api.VolumeSource{HostDir: &api.HostDir{"/mnt/path2"}}},
		{Name: "abc-123", Source: &api.VolumeSource{HostDir: &api.HostDir{"/mnt/path3"}}},
		{Name: "empty", Source: &api.VolumeSource{EmptyDir: &api.EmptyDir{}}},
		{Name: "gcepd", Source: &api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDisk{"my-PD", "ext4", 1, false}}},
	}
	names, errs := validateVolumes(successCase)
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
	if len(names) != 5 || !names.HasAll("abc", "123", "abc-123", "empty", "gcepd") {
		t.Errorf("wrong names result: %v", names)
	}

	errorCases := map[string]struct {
		V []api.Volume
		T errors.ValidationErrorType
		F string
	}{
		"zero-length name":     {[]api.Volume{{Name: ""}}, errors.ValidationErrorTypeRequired, "[0].name"},
		"name > 63 characters": {[]api.Volume{{Name: strings.Repeat("a", 64)}}, errors.ValidationErrorTypeInvalid, "[0].name"},
		"name not a DNS label": {[]api.Volume{{Name: "a.b.c"}}, errors.ValidationErrorTypeInvalid, "[0].name"},
		"name not unique":      {[]api.Volume{{Name: "abc"}, {Name: "abc"}}, errors.ValidationErrorTypeDuplicate, "[1].name"},
	}
	for k, v := range errorCases {
		_, errs := validateVolumes(v.V)
		if len(errs) == 0 {
			t.Errorf("expected failure %s for %v", k, v.V)
			continue
		}
		for i := range errs {
			if errs[i].(errors.ValidationError).Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].(errors.ValidationError).Field != v.F {
				t.Errorf("%s: expected errors to have field %s: %v", k, v.F, errs[i])
			}
		}
	}
}

func TestValidatePorts(t *testing.T) {
	successCase := []api.Port{
		{Name: "abc", ContainerPort: 80, HostPort: 80, Protocol: "TCP"},
		{Name: "123", ContainerPort: 81, HostPort: 81},
		{Name: "easy", ContainerPort: 82, Protocol: "TCP"},
		{Name: "as", ContainerPort: 83, Protocol: "UDP"},
		{Name: "do-re-me", ContainerPort: 84},
		{Name: "baby-you-and-me", ContainerPort: 82, Protocol: "tcp"},
		{ContainerPort: 85},
	}
	if errs := validatePorts(successCase); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	nonCanonicalCase := []api.Port{
		{ContainerPort: 80},
	}
	if errs := validatePorts(nonCanonicalCase); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
	if nonCanonicalCase[0].HostPort != 0 || nonCanonicalCase[0].Protocol != "TCP" {
		t.Errorf("expected default values: %+v", nonCanonicalCase[0])
	}

	errorCases := map[string]struct {
		P []api.Port
		T errors.ValidationErrorType
		F string
	}{
		"name > 63 characters": {[]api.Port{{Name: strings.Repeat("a", 64), ContainerPort: 80}}, errors.ValidationErrorTypeInvalid, "[0].name"},
		"name not a DNS label": {[]api.Port{{Name: "a.b.c", ContainerPort: 80}}, errors.ValidationErrorTypeInvalid, "[0].name"},
		"name not unique": {[]api.Port{
			{Name: "abc", ContainerPort: 80},
			{Name: "abc", ContainerPort: 81},
		}, errors.ValidationErrorTypeDuplicate, "[1].name"},
		"zero container port":    {[]api.Port{{ContainerPort: 0}}, errors.ValidationErrorTypeRequired, "[0].containerPort"},
		"invalid container port": {[]api.Port{{ContainerPort: 65536}}, errors.ValidationErrorTypeInvalid, "[0].containerPort"},
		"invalid host port":      {[]api.Port{{ContainerPort: 80, HostPort: 65536}}, errors.ValidationErrorTypeInvalid, "[0].hostPort"},
		"invalid protocol":       {[]api.Port{{ContainerPort: 80, Protocol: "ICMP"}}, errors.ValidationErrorTypeNotSupported, "[0].protocol"},
	}
	for k, v := range errorCases {
		errs := validatePorts(v.P)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			if errs[i].(errors.ValidationError).Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].(errors.ValidationError).Field != v.F {
				t.Errorf("%s: expected errors to have field %s: %v", k, v.F, errs[i])
			}
		}
	}
}

func TestValidateEnv(t *testing.T) {
	successCase := []api.EnvVar{
		{Name: "abc", Value: "value"},
		{Name: "ABC", Value: "value"},
		{Name: "AbC_123", Value: "value"},
		{Name: "abc", Value: ""},
	}
	if errs := validateEnv(successCase); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string][]api.EnvVar{
		"zero-length name":        {{Name: ""}},
		"name not a C identifier": {{Name: "a.b.c"}},
	}
	for k, v := range errorCases {
		if errs := validateEnv(v); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateVolumeMounts(t *testing.T) {
	volumes := util.NewStringSet("abc", "123", "abc-123")

	successCase := []api.VolumeMount{
		{Name: "abc", MountPath: "/foo"},
		{Name: "123", MountPath: "/foo"},
		{Name: "abc-123", MountPath: "/bar"},
	}
	if errs := validateVolumeMounts(successCase, volumes); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string][]api.VolumeMount{
		"empty name":      {{Name: "", MountPath: "/foo"}},
		"name not found":  {{Name: "", MountPath: "/foo"}},
		"empty mountpath": {{Name: "abc", MountPath: ""}},
	}
	for k, v := range errorCases {
		if errs := validateVolumeMounts(v, volumes); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateContainers(t *testing.T) {
	volumes := util.StringSet{}
	capabilities.SetForTests(capabilities.Capabilities{
		AllowPrivileged: true,
	})

	successCase := []api.Container{
		{Name: "abc", Image: "image"},
		{Name: "123", Image: "image"},
		{Name: "abc-123", Image: "image"},
		{
			Name:  "life-123",
			Image: "image",
			Lifecycle: &api.Lifecycle{
				PreStop: &api.Handler{
					Exec: &api.ExecAction{Command: []string{"ls", "-l"}},
				},
			},
		},
		{Name: "abc-1234", Image: "image", Privileged: true},
	}
	if errs := validateContainers(successCase, volumes); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	capabilities.SetForTests(capabilities.Capabilities{
		AllowPrivileged: false,
	})
	errorCases := map[string][]api.Container{
		"zero-length name":     {{Name: "", Image: "image"}},
		"name > 63 characters": {{Name: strings.Repeat("a", 64), Image: "image"}},
		"name not a DNS label": {{Name: "a.b.c", Image: "image"}},
		"name not unique": {
			{Name: "abc", Image: "image"},
			{Name: "abc", Image: "image"},
		},
		"zero-length image": {{Name: "abc", Image: ""}},
		"host port not unique": {
			{Name: "abc", Image: "image", Ports: []api.Port{{ContainerPort: 80, HostPort: 80}}},
			{Name: "def", Image: "image", Ports: []api.Port{{ContainerPort: 81, HostPort: 80}}},
		},
		"invalid env var name": {
			{Name: "abc", Image: "image", Env: []api.EnvVar{{Name: "ev.1"}}},
		},
		"unknown volume name": {
			{Name: "abc", Image: "image", VolumeMounts: []api.VolumeMount{{Name: "anything", MountPath: "/foo"}}},
		},
		"invalid lifecycle, no exec command.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &api.Lifecycle{
					PreStop: &api.Handler{
						Exec: &api.ExecAction{},
					},
				},
			},
		},
		"invalid lifecycle, no http path.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &api.Lifecycle{
					PreStop: &api.Handler{
						HTTPGet: &api.HTTPGetAction{},
					},
				},
			},
		},
		"invalid lifecycle, no action.": {
			{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &api.Lifecycle{
					PreStop: &api.Handler{},
				},
			},
		},
		"privilege disabled": {
			{Name: "abc", Image: "image", Privileged: true},
		},
	}
	for k, v := range errorCases {
		if errs := validateContainers(v, volumes); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateRestartPolicy(t *testing.T) {
	successCases := []api.RestartPolicy{
		{},
		{Always: &api.RestartPolicyAlways{}},
		{OnFailure: &api.RestartPolicyOnFailure{}},
		{Never: &api.RestartPolicyNever{}},
	}
	for _, policy := range successCases {
		if errs := validateRestartPolicy(&policy); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []api.RestartPolicy{
		{Always: &api.RestartPolicyAlways{}, Never: &api.RestartPolicyNever{}},
		{Never: &api.RestartPolicyNever{}, OnFailure: &api.RestartPolicyOnFailure{}},
	}
	for k, policy := range errorCases {
		if errs := validateRestartPolicy(&policy); len(errs) == 0 {
			t.Errorf("expected failure for %d", k)
		}
	}

	noPolicySpecified := api.RestartPolicy{}
	errs := validateRestartPolicy(&noPolicySpecified)
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
	if noPolicySpecified.Always == nil {
		t.Errorf("expected Always policy specified")
	}

}

func TestValidateManifest(t *testing.T) {
	successCases := []api.ContainerManifest{
		{Version: "v1beta1", ID: "abc"},
		{Version: "v1beta2", ID: "123"},
		{Version: "V1BETA1", ID: "abc.123.do-re-mi"},
		{
			Version: "v1beta1",
			ID:      "abc",
			Volumes: []api.Volume{{Name: "vol1", Source: &api.VolumeSource{HostDir: &api.HostDir{"/mnt/vol1"}}},
				{Name: "vol2", Source: &api.VolumeSource{HostDir: &api.HostDir{"/mnt/vol2"}}}},
			Containers: []api.Container{
				{
					Name:       "abc",
					Image:      "image",
					Command:    []string{"foo", "bar"},
					WorkingDir: "/tmp",
					Memory:     1,
					CPU:        1,
					Ports: []api.Port{
						{Name: "p1", ContainerPort: 80, HostPort: 8080},
						{Name: "p2", ContainerPort: 81},
						{ContainerPort: 82},
					},
					Env: []api.EnvVar{
						{Name: "ev1", Value: "val1"},
						{Name: "ev2", Value: "val2"},
						{Name: "EV3", Value: "val3"},
					},
					VolumeMounts: []api.VolumeMount{
						{Name: "vol1", MountPath: "/foo"},
						{Name: "vol1", MountPath: "/bar"},
					},
				},
			},
		},
	}
	for _, manifest := range successCases {
		if errs := ValidateManifest(&manifest); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]api.ContainerManifest{
		"empty version":   {Version: "", ID: "abc"},
		"invalid version": {Version: "bogus", ID: "abc"},
		"invalid volume name": {
			Version: "v1beta1",
			ID:      "abc",
			Volumes: []api.Volume{{Name: "vol.1"}},
		},
		"invalid container name": {
			Version:    "v1beta1",
			ID:         "abc",
			Containers: []api.Container{{Name: "ctr.1", Image: "image"}},
		},
	}
	for k, v := range errorCases {
		if errs := ValidateManifest(&v); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidatePod(t *testing.T) {
	errs := ValidatePod(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "foo", Namespace: api.NamespaceDefault,
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Version: "v1beta1",
				ID:      "abc",
				RestartPolicy: api.RestartPolicy{
					Always: &api.RestartPolicyAlways{},
				},
			},
		},
	})
	if len(errs) != 0 {
		t.Errorf("Unexpected non-zero error list: %#v", errs)
	}
	errs = ValidatePod(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{Version: "v1beta1", ID: "abc"},
		},
	})
	if len(errs) != 0 {
		t.Errorf("Unexpected non-zero error list: %#v", errs)
	}

	errs = ValidatePod(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "foo", Namespace: api.NamespaceDefault,
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Version: "v1beta1",
				ID:      "abc",
				RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{},
					Never: &api.RestartPolicyNever{}},
			},
		},
	})
	if len(errs) != 1 {
		t.Errorf("Unexpected error list: %#v", errs)
	}
	errs = ValidatePod(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
			Labels: map[string]string{
				"123cantbeginwithnumber":          "bar", //invalid
				"NoUppercase123":                  "bar", //invalid
				"nospecialchars^=@":               "bar", //invalid
				"cantendwithadash-":               "bar", //invalid
				"rfc952-mustbe24charactersorless": "bar", //invalid
				"rfc952-dash-nodots-lower":        "bar", //good label
				"rfc952-24chars-orless":           "bar", //good label
			},
		},
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{Version: "v1beta1", ID: "abc"},
		},
	})
	if len(errs) != 5 {
		t.Errorf("Unexpected non-zero error list: %#v", errs)
	}
}

func TestValidatePodUpdate(t *testing.T) {
	tests := []struct {
		a       api.Pod
		b       api.Pod
		isValid bool
		test    string
	}{
		{api.Pod{}, api.Pod{}, true, "nothing"},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "bar"},
			},
			false,
			"ids",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"foo": "bar",
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
					Labels: map[string]string{
						"bar": "foo",
					},
				},
			},
			true,
			"labels",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "foo:V1",
							},
						},
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "foo:V2",
							},
							{
								Image: "bar:V2",
							},
						},
					},
				},
			},
			false,
			"more containers",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "foo:V1",
							},
						},
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "foo:V2",
							},
						},
					},
				},
			},
			true,
			"image change",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "foo:V1",
								CPU:   100,
							},
						},
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "foo:V2",
								CPU:   1000,
							},
						},
					},
				},
			},
			false,
			"cpu change",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "foo:V1",
								Ports: []api.Port{
									{HostPort: 8080, ContainerPort: 80},
								},
							},
						},
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: "foo:V2",
								Ports: []api.Port{
									{HostPort: 8000, ContainerPort: 80},
								},
							},
						},
					},
				},
			},
			false,
			"port change",
		},
	}

	for _, test := range tests {
		errs := ValidatePodUpdate(&test.a, &test.b)
		if test.isValid {
			if len(errs) != 0 {
				t.Errorf("unexpected invalid: %s %v, %v", test.test, test.a, test.b)
			}
		} else {
			if len(errs) == 0 {
				t.Errorf("unexpected valid: %s %v, %v", test.test, test.a, test.b)
			}
		}
	}
}

func TestValidateService(t *testing.T) {
	testCases := []struct {
		name     string
		svc      api.Service
		existing api.ServiceList
		numErrs  int
	}{
		{
			name: "missing id",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Port:     8675,
					Selector: map[string]string{"foo": "bar"},
				},
			},
			// Should fail because the ID is missing.
			numErrs: 1,
		},
		{
			name: "missing namespace",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.ServiceSpec{
					Port:     8675,
					Selector: map[string]string{"foo": "bar"},
				},
			},
			// Should fail because the Namespace is missing.
			numErrs: 1,
		},
		{
			name: "invalid id",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "123abc", Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Port:     8675,
					Selector: map[string]string{"foo": "bar"},
				},
			},
			// Should fail because the ID is invalid.
			numErrs: 1,
		},
		{
			name: "missing port",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "abc123", Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Selector: map[string]string{"foo": "bar"},
				},
			},
			// Should fail because the port number is missing/invalid.
			numErrs: 1,
		},
		{
			name: "invalid port",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "abc123", Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Port:     66536,
					Selector: map[string]string{"foo": "bar"},
				},
			},
			// Should fail because the port number is invalid.
			numErrs: 1,
		},
		{
			name: "invalid protocol",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "abc123", Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Port:     8675,
					Selector: map[string]string{"foo": "bar"},
					Protocol: "INVALID",
				},
			},
			// Should fail because the protocol is invalid.
			numErrs: 1,
		},
		{
			name: "missing selector",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Port: 8675,
				},
			},
			// Should fail because the selector is missing.
			numErrs: 1,
		},
		{
			name: "valid 1",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "abc123", Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Port:     8675,
					Selector: map[string]string{"foo": "bar"},
					Protocol: "TCP",
				},
			},
			numErrs: 0,
		},
		{
			name: "valid 2",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "abc123", Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Port:     8675,
					Selector: map[string]string{"foo": "bar"},
					Protocol: "UDP",
				},
			},
			numErrs: 0,
		},
		{
			name: "valid 3",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "abc123", Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Port:     8675,
					Selector: map[string]string{"foo": "bar"},
				},
			},
			numErrs: 0,
		},
		{
			name: "invalid port in use",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "abc123", Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Port: 80,
					CreateExternalLoadBalancer: true,
					Selector:                   map[string]string{"foo": "bar"},
				},
			},
			existing: api.ServiceList{
				Items: []api.Service{
					{Spec: api.ServiceSpec{Port: 80, CreateExternalLoadBalancer: true}},
				},
			},
			numErrs: 1,
		},
		{
			name: "same port in use, but not external",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "abc123", Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Port: 80,
					CreateExternalLoadBalancer: true,
					Selector:                   map[string]string{"foo": "bar"},
				},
			},
			existing: api.ServiceList{
				Items: []api.Service{
					{Spec: api.ServiceSpec{Port: 80}},
				},
			},
			numErrs: 0,
		},
		{
			name: "same port in use, but not external on input",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "abc123", Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Port:     80,
					Selector: map[string]string{"foo": "bar"},
				},
			},
			existing: api.ServiceList{
				Items: []api.Service{
					{Spec: api.ServiceSpec{Port: 80, CreateExternalLoadBalancer: true}},
				},
			},
			numErrs: 0,
		},
		{
			name: "same port in use, but neither external",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{Name: "abc123", Namespace: api.NamespaceDefault},
				Spec: api.ServiceSpec{
					Port:     80,
					Selector: map[string]string{"foo": "bar"},
				},
			},
			existing: api.ServiceList{
				Items: []api.Service{
					{Spec: api.ServiceSpec{Port: 80}},
				},
			},
			numErrs: 0,
		},
		{
			name: "invalid label",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:      "abc123",
					Namespace: api.NamespaceDefault,
					Labels: map[string]string{
						"NoUppercaseOrSpecialCharsLike=Equals": "bar",
					},
				},
				Spec: api.ServiceSpec{
					Port:     8675,
					Selector: map[string]string{"foo": "bar", "NoUppercaseOrSpecialCharsLike=Equals": "bar"},
				},
			},
			numErrs: 2,
		},
	}

	for _, tc := range testCases {
		registry := registrytest.NewServiceRegistry()
		registry.List = tc.existing
		errs := ValidateService(&tc.svc, registry, api.NewDefaultContext())
		if len(errs) != tc.numErrs {
			t.Errorf("Unexpected error list for case %q: %+v", tc.name, errs)
		}
	}

	svc := api.Service{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault},
		Spec: api.ServiceSpec{
			Port:     8675,
			Selector: map[string]string{"foo": "bar"},
		},
	}
	errs := ValidateService(&svc, registrytest.NewServiceRegistry(), api.NewDefaultContext())
	if len(errs) != 0 {
		t.Errorf("Unexpected non-zero error list: %#v", errs)
	}
	if svc.Spec.Protocol != "TCP" {
		t.Errorf("Expected default protocol of 'TCP': %#v", errs)
	}
}

func TestValidateReplicationController(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	validPodTemplate := api.PodTemplate{
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Version: "v1beta1",
			},
		},
		Labels: validSelector,
	}
	invalidVolumePodTemplate := api.PodTemplate{
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Version: "v1beta1",
				Volumes: []api.Volume{{Name: "gcepd", Source: &api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDisk{"my-PD", "ext4", 1, false}}}},
			},
		},
	}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	invalidPodTemplate := api.PodTemplate{
		DesiredState: api.PodState{
			Manifest: api.ContainerManifest{
				Version: "v1beta1",
			},
		},
		Labels: invalidSelector,
	}
	successCases := []api.ReplicationController{
		{
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			DesiredState: api.ReplicationControllerState{
				ReplicaSelector: validSelector,
				PodTemplate:     validPodTemplate,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc-123", Namespace: api.NamespaceDefault},
			DesiredState: api.ReplicationControllerState{
				ReplicaSelector: validSelector,
				PodTemplate:     validPodTemplate,
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateReplicationController(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]api.ReplicationController{
		"zero-length ID": {
			ObjectMeta: api.ObjectMeta{Name: "", Namespace: api.NamespaceDefault},
			DesiredState: api.ReplicationControllerState{
				ReplicaSelector: validSelector,
				PodTemplate:     validPodTemplate,
			},
		},
		"missing-namespace": {
			ObjectMeta: api.ObjectMeta{Name: "abc-123"},
			DesiredState: api.ReplicationControllerState{
				ReplicaSelector: validSelector,
				PodTemplate:     validPodTemplate,
			},
		},
		"empty selector": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			DesiredState: api.ReplicationControllerState{
				PodTemplate: validPodTemplate,
			},
		},
		"selector_doesnt_match": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			DesiredState: api.ReplicationControllerState{
				ReplicaSelector: map[string]string{"foo": "bar"},
				PodTemplate:     validPodTemplate,
			},
		},
		"invalid manifest": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			DesiredState: api.ReplicationControllerState{
				ReplicaSelector: validSelector,
			},
		},
		"read-write presistent disk": {
			ObjectMeta: api.ObjectMeta{Name: "abc"},
			DesiredState: api.ReplicationControllerState{
				ReplicaSelector: validSelector,
				PodTemplate:     invalidVolumePodTemplate,
			},
		},
		"negative_replicas": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			DesiredState: api.ReplicationControllerState{
				Replicas:        -1,
				ReplicaSelector: validSelector,
			},
		},
		"invalid_label": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			DesiredState: api.ReplicationControllerState{
				ReplicaSelector: validSelector,
				PodTemplate:     validPodTemplate,
			},
		},
		"invalid_label 2": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
				Labels: map[string]string{
					"NoUppercaseOrSpecialCharsLike=Equals": "bar",
				},
			},
			DesiredState: api.ReplicationControllerState{
				PodTemplate: invalidPodTemplate,
			},
		},
	}
	for k, v := range errorCases {
		errs := ValidateReplicationController(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			field := errs[i].(errors.ValidationError).Field
			if !strings.HasPrefix(field, "desiredState.podTemplate.") &&
				field != "name" &&
				field != "namespace" &&
				field != "desiredState.replicaSelector" &&
				field != "GCEPersistentDisk.ReadOnly" &&
				field != "desiredState.replicas" &&
				field != "desiredState.label" &&
				field != "label" {
				t.Errorf("%s: missing prefix for: %v", k, errs[i])
			}
		}
	}
}

func TestValidateBoundPodNoName(t *testing.T) {
	errorCases := map[string]api.BoundPod{
		// manifest is tested in api/validation_test.go, ensure it is invoked
		"empty version": {ObjectMeta: api.ObjectMeta{Name: "test"}, Spec: api.PodSpec{Containers: []api.Container{{Name: ""}}}},

		// Name
		"zero-length name":         {ObjectMeta: api.ObjectMeta{Name: ""}},
		"name > 255 characters":    {ObjectMeta: api.ObjectMeta{Name: strings.Repeat("a", 256)}},
		"name not a DNS subdomain": {ObjectMeta: api.ObjectMeta{Name: "a.b.c."}},
		"name with underscore":     {ObjectMeta: api.ObjectMeta{Name: "a_b_c"}},
	}
	for k, v := range errorCases {
		if errs := ValidateBoundPod(&v); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}
