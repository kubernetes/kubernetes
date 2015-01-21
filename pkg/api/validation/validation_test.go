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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	utilerrors "github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
)

func expectPrefix(t *testing.T, prefix string, errs errors.ValidationErrorList) {
	for i := range errs {
		if f, p := errs[i].(*errors.ValidationError).Field, prefix; !strings.HasPrefix(f, p) {
			t.Errorf("expected prefix '%s' for field '%s' (%v)", p, f, errs[i])
		}
	}
}

func TestValidateLabels(t *testing.T) {
	successCases := []map[string]string{
		{"simple": "bar"},
		{"now-with-dashes": "bar"},
		{"1-starts-with-num": "bar"},
		{"1234": "bar"},
		{"simple/simple": "bar"},
		{"now-with-dashes/simple": "bar"},
		{"now-with-dashes/now-with-dashes": "bar"},
		{"now.with.dots/simple": "bar"},
		{"now-with.dashes-and.dots/simple": "bar"},
		{"1-num.2-num/3-num": "bar"},
		{"1234/5678": "bar"},
		{"1.2.3.4/5678": "bar"},
	}
	for i := range successCases {
		errs := ValidateLabels(successCases[i], "field")
		if len(errs) != 0 {
			t.Errorf("case[%d] expected success, got %#v", i, errs)
		}
	}

	errorCases := []map[string]string{
		{"NoUppercase123": "bar"},
		{"nospecialchars^=@": "bar"},
		{"cantendwithadash-": "bar"},
		{"only/one/slash": "bar"},
		{strings.Repeat("a", 254): "bar"},
	}
	for i := range errorCases {
		errs := ValidateLabels(errorCases[i], "field")
		if len(errs) != 1 {
			t.Errorf("case[%d] expected failure", i)
		}
	}
}

func TestValidateVolumes(t *testing.T) {
	successCase := []api.Volume{
		{Name: "abc"},
		{Name: "123", Source: api.VolumeSource{HostPath: &api.HostPath{"/mnt/path2"}}},
		{Name: "abc-123", Source: api.VolumeSource{HostPath: &api.HostPath{"/mnt/path3"}}},
		{Name: "empty", Source: api.VolumeSource{EmptyDir: &api.EmptyDir{}}},
		{Name: "gcepd", Source: api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDisk{"my-PD", "ext4", 1, false}}},
		{Name: "gitrepo", Source: api.VolumeSource{GitRepo: &api.GitRepo{"my-repo", "hashstring"}}},
	}
	names, errs := validateVolumes(successCase)
	if len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
	if len(names) != 6 || !names.HasAll("abc", "123", "abc-123", "empty", "gcepd", "gitrepo") {
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
			if errs[i].(*errors.ValidationError).Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].(*errors.ValidationError).Field != v.F {
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
			if errs[i].(*errors.ValidationError).Type != v.T {
				t.Errorf("%s: expected errors to have type %s: %v", k, v.T, errs[i])
			}
			if errs[i].(*errors.ValidationError).Field != v.F {
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

func TestValidatePullPolicy(t *testing.T) {
	type T struct {
		Container      api.Container
		ExpectedPolicy api.PullPolicy
	}
	testCases := map[string]T{
		"NotPresent1": {
			api.Container{Name: "abc", Image: "image:latest", ImagePullPolicy: "PullIfNotPresent"},
			api.PullIfNotPresent,
		},
		"NotPresent2": {
			api.Container{Name: "abc1", Image: "image", ImagePullPolicy: "PullIfNotPresent"},
			api.PullIfNotPresent,
		},
		"Always1": {
			api.Container{Name: "123", Image: "image:latest", ImagePullPolicy: "PullAlways"},
			api.PullAlways,
		},
		"Always2": {
			api.Container{Name: "1234", Image: "image", ImagePullPolicy: "PullAlways"},
			api.PullAlways,
		},
		"Never1": {
			api.Container{Name: "abc-123", Image: "image:latest", ImagePullPolicy: "PullNever"},
			api.PullNever,
		},
		"Never2": {
			api.Container{Name: "abc-1234", Image: "image", ImagePullPolicy: "PullNever"},
			api.PullNever,
		},
		"DefaultToNotPresent":  {api.Container{Name: "notPresent", Image: "image"}, api.PullIfNotPresent},
		"DefaultToNotPresent2": {api.Container{Name: "notPresent1", Image: "image:sometag"}, api.PullIfNotPresent},
		"DefaultToAlways1":     {api.Container{Name: "always", Image: "image:latest"}, api.PullAlways},
		"DefaultToAlways2":     {api.Container{Name: "always", Image: "foo.bar.com:5000/my/image:latest"}, api.PullAlways},
	}
	for k, v := range testCases {
		ctr := &v.Container
		errs := validatePullPolicyWithDefault(ctr)
		if len(errs) != 0 {
			t.Errorf("case[%s] expected success, got %#v", k, errs)
		}
		if ctr.ImagePullPolicy != v.ExpectedPolicy {
			t.Errorf("case[%s] expected policy %v, got %v", k, v.ExpectedPolicy, ctr.ImagePullPolicy)
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

func TestValidateDNSPolicy(t *testing.T) {
	successCases := []api.DNSPolicy{api.DNSClusterFirst, api.DNSDefault, api.DNSPolicy("")}
	for _, policy := range successCases {
		if errs := validateDNSPolicy(&policy); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []api.DNSPolicy{api.DNSPolicy("invalid")}
	for _, policy := range errorCases {
		if errs := validateDNSPolicy(&policy); len(errs) == 0 {
			t.Errorf("expected failure for %v", policy)
		}
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
			Volumes: []api.Volume{{Name: "vol1", Source: api.VolumeSource{HostPath: &api.HostPath{"/mnt/vol1"}}},
				{Name: "vol2", Source: api.VolumeSource{HostPath: &api.HostPath{"/mnt/vol2"}}}},
			Containers: []api.Container{
				{
					Name:       "abc",
					Image:      "image",
					Command:    []string{"foo", "bar"},
					WorkingDir: "/tmp",
					Memory:     resource.MustParse("1"),
					CPU:        resource.MustParse("1"),
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

func TestValidatePodSpec(t *testing.T) {
	successCases := []api.PodSpec{
		{}, // empty is valid, if not very useful */
		{ // Populate basic fields, leave defaults for most.
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
		{ // Populate all fields.
			Volumes: []api.Volume{
				{Name: "vol"},
			},
			Containers:    []api.Container{{Name: "ctr", Image: "image"}},
			RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
			DNSPolicy:     api.DNSClusterFirst,
			NodeSelector: map[string]string{
				"key": "value",
			},
			Host: "foobar",
		},
	}
	for i := range successCases {
		if errs := ValidatePodSpec(&successCases[i]); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	failureCases := map[string]api.PodSpec{
		"bad volume": {
			Volumes: []api.Volume{{}},
		},
		"bad container": {
			Containers: []api.Container{{}},
		},
		"bad DNS policy": {
			DNSPolicy: api.DNSPolicy("invalid"),
		},
	}
	for k, v := range failureCases {
		if errs := ValidatePodSpec(&v); len(errs) == 0 {
			t.Errorf("expected failure for %q", k)
		}
	}

	defaultPod := api.PodSpec{} // all empty fields
	if errs := ValidatePodSpec(&defaultPod); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
	if util.AllPtrFieldsNil(defaultPod.RestartPolicy) {
		t.Errorf("expected a default RestartPolicy")
	}
	if defaultPod.DNSPolicy == "" {
		t.Errorf("expected a default DNSPolicy")
	}
}

func TestValidatePod(t *testing.T) {
	successCases := []api.Pod{
		{ // Mostly empty.
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "ns"},
		},
		{ // Basic fields.
			ObjectMeta: api.ObjectMeta{Name: "123", Namespace: "ns"},
			Spec: api.PodSpec{
				Volumes:    []api.Volume{{Name: "vol"}},
				Containers: []api.Container{{Name: "ctr", Image: "image"}},
			},
		},
		{ // Just about everything.
			ObjectMeta: api.ObjectMeta{Name: "abc.123.do-re-mi", Namespace: "ns"},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{Name: "vol"},
				},
				Containers:    []api.Container{{Name: "ctr", Image: "image"}},
				RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
				DNSPolicy:     api.DNSClusterFirst,
				NodeSelector: map[string]string{
					"key": "value",
				},
				Host: "foobar",
			},
		},
	}
	for _, pod := range successCases {
		if errs := ValidatePod(&pod); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]api.Pod{
		"bad name":      {ObjectMeta: api.ObjectMeta{Name: "", Namespace: "ns"}},
		"bad namespace": {ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: ""}},
		"bad spec": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "ns"},
			Spec: api.PodSpec{
				Containers: []api.Container{{}},
			},
		},
	}
	for k, v := range errorCases {
		if errs := ValidatePod(&v); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
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
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V1",
						},
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
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
			false,
			"more containers",
		},
		{
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V1",
						},
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V2",
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
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V1",
							CPU:   resource.MustParse("100m"),
						},
					},
				},
			},
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Image: "foo:V2",
							CPU:   resource.MustParse("1000m"),
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
				Spec: api.PodSpec{
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
			api.Pod{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec: api.PodSpec{
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

func TestValidateBoundPods(t *testing.T) {
	successCases := []api.BoundPod{
		{ // Mostly empty.
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "ns"},
		},
		{ // Basic fields.
			ObjectMeta: api.ObjectMeta{Name: "123", Namespace: "ns"},
			Spec: api.PodSpec{
				Volumes:    []api.Volume{{Name: "vol"}},
				Containers: []api.Container{{Name: "ctr", Image: "image"}},
			},
		},
		{ // Just about everything.
			ObjectMeta: api.ObjectMeta{Name: "abc.123.do-re-mi", Namespace: "ns"},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{Name: "vol"},
				},
				Containers:    []api.Container{{Name: "ctr", Image: "image"}},
				RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
				DNSPolicy:     api.DNSClusterFirst,
				NodeSelector: map[string]string{
					"key": "value",
				},
				Host: "foobar",
			},
		},
	}
	for _, pod := range successCases {
		if errs := ValidateBoundPod(&pod); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]api.Pod{
		"bad name":      {ObjectMeta: api.ObjectMeta{Name: "", Namespace: "ns"}},
		"bad namespace": {ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: ""}},
		"bad spec": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: "ns"},
			Spec: api.PodSpec{
				Containers: []api.Container{{}},
			},
		},
	}
	for k, v := range errorCases {
		if errs := ValidatePod(&v); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
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
			// Should be ok because the selector is missing.
			numErrs: 0,
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
					{
						ObjectMeta: api.ObjectMeta{Name: "def123", Namespace: api.NamespaceDefault},
						Spec:       api.ServiceSpec{Port: 80, CreateExternalLoadBalancer: true},
					},
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
					{
						ObjectMeta: api.ObjectMeta{Name: "def123", Namespace: api.NamespaceDefault},
						Spec:       api.ServiceSpec{Port: 80},
					},
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
					{
						ObjectMeta: api.ObjectMeta{Name: "def123", Namespace: api.NamespaceDefault},
						Spec:       api.ServiceSpec{Port: 80, CreateExternalLoadBalancer: true},
					},
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
					{
						ObjectMeta: api.ObjectMeta{Name: "def123", Namespace: api.NamespaceDefault},
						Spec:       api.ServiceSpec{Port: 80},
					},
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
					Port: 8675,
				},
			},
			numErrs: 1,
		},
		{
			name: "invalid selector",
			svc: api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:      "abc123",
					Namespace: api.NamespaceDefault,
				},
				Spec: api.ServiceSpec{
					Port:     8675,
					Selector: map[string]string{"foo": "bar", "NoUppercaseOrSpecialCharsLike=Equals": "bar"},
				},
			},
			numErrs: 1,
		},
	}

	for _, tc := range testCases {
		registry := registrytest.NewServiceRegistry()
		registry.List = tc.existing
		errs := ValidateService(&tc.svc, registry, api.NewDefaultContext())
		if len(errs) != tc.numErrs {
			t.Errorf("Unexpected error list for case %q: %v", tc.name, utilerrors.NewAggregate(errs))
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
		Spec: api.PodTemplateSpec{
			ObjectMeta: api.ObjectMeta{
				Labels: validSelector,
			},
		},
	}
	invalidVolumePodTemplate := api.PodTemplate{
		Spec: api.PodTemplateSpec{
			Spec: api.PodSpec{
				Volumes: []api.Volume{{Name: "gcepd", Source: api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDisk{"my-PD", "ext4", 1, false}}}},
			},
		},
	}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	invalidPodTemplate := api.PodTemplate{
		Spec: api.PodTemplateSpec{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicy{
					Always: &api.RestartPolicyAlways{},
				},
			},
			ObjectMeta: api.ObjectMeta{
				Labels: invalidSelector,
			},
		},
	}
	successCases := []api.ReplicationController{
		{
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Spec,
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc-123", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Spec,
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
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Spec,
			},
		},
		"missing-namespace": {
			ObjectMeta: api.ObjectMeta{Name: "abc-123"},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Spec,
			},
		},
		"empty selector": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Template: &validPodTemplate.Spec,
			},
		},
		"selector_doesnt_match": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Selector: map[string]string{"foo": "bar"},
				Template: &validPodTemplate.Spec,
			},
		},
		"invalid manifest": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
			},
		},
		"read-write persistent disk": {
			ObjectMeta: api.ObjectMeta{Name: "abc"},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &invalidVolumePodTemplate.Spec,
			},
		},
		"negative_replicas": {
			ObjectMeta: api.ObjectMeta{Name: "abc", Namespace: api.NamespaceDefault},
			Spec: api.ReplicationControllerSpec{
				Replicas: -1,
				Selector: validSelector,
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
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &validPodTemplate.Spec,
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
			Spec: api.ReplicationControllerSpec{
				Template: &invalidPodTemplate.Spec,
			},
		},
		"invalid restart policy 1": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
			},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &api.PodTemplateSpec{
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicy{
							OnFailure: &api.RestartPolicyOnFailure{},
						},
					},
					ObjectMeta: api.ObjectMeta{
						Labels: validSelector,
					},
				},
			},
		},
		"invalid restart policy 2": {
			ObjectMeta: api.ObjectMeta{
				Name:      "abc-123",
				Namespace: api.NamespaceDefault,
			},
			Spec: api.ReplicationControllerSpec{
				Selector: validSelector,
				Template: &api.PodTemplateSpec{
					Spec: api.PodSpec{
						RestartPolicy: api.RestartPolicy{
							Never: &api.RestartPolicyNever{},
						},
					},
					ObjectMeta: api.ObjectMeta{
						Labels: validSelector,
					},
				},
			},
		},
	}
	for k, v := range errorCases {
		errs := ValidateReplicationController(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			field := errs[i].(*errors.ValidationError).Field
			if !strings.HasPrefix(field, "spec.template.") &&
				field != "name" &&
				field != "namespace" &&
				field != "spec.selector" &&
				field != "spec.template" &&
				field != "GCEPersistentDisk.ReadOnly" &&
				field != "spec.replicas" &&
				field != "spec.template.labels" &&
				field != "labels" {
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

func TestValidateMinion(t *testing.T) {
	validSelector := map[string]string{"a": "b"}
	invalidSelector := map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "b"}
	successCases := []api.Node{
		{
			ObjectMeta: api.ObjectMeta{
				Name:   "abc",
				Labels: validSelector,
			},
			Status: api.NodeStatus{
				HostIP: "something",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "abc"},
			Status: api.NodeStatus{
				HostIP: "something",
			},
		},
	}
	for _, successCase := range successCases {
		if errs := ValidateMinion(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := map[string]api.Node{
		"zero-length Name": {
			ObjectMeta: api.ObjectMeta{
				Name:   "",
				Labels: validSelector,
			},
			Status: api.NodeStatus{
				HostIP: "something",
			},
		},
		"invalid-labels": {
			ObjectMeta: api.ObjectMeta{
				Name:   "abc-123",
				Labels: invalidSelector,
			},
		},
	}
	for k, v := range errorCases {
		errs := ValidateMinion(&v)
		if len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
		for i := range errs {
			field := errs[i].(*errors.ValidationError).Field
			if field != "name" &&
				field != "labels" {
				t.Errorf("%s: missing prefix for: %v", k, errs[i])
			}
		}
	}
}

func TestValidateMinionUpdate(t *testing.T) {
	tests := []struct {
		oldMinion api.Node
		minion    api.Node
		valid     bool
	}{
		{api.Node{}, api.Node{}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo"}},
			api.Node{
				ObjectMeta: api.ObjectMeta{
					Name: "bar"},
			}, false},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "bar"},
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "foo"},
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"foo": "baz"},
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Spec: api.NodeSpec{
				Capacity: api.ResourceList{
					api.ResourceCPU:    resource.MustParse("10000"),
					api.ResourceMemory: resource.MustParse("100"),
				},
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name: "foo",
			},
			Spec: api.NodeSpec{
				Capacity: api.ResourceList{
					api.ResourceCPU:    resource.MustParse("100"),
					api.ResourceMemory: resource.MustParse("10000"),
				},
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "foo"},
			},
			Spec: api.NodeSpec{
				Capacity: api.ResourceList{
					api.ResourceCPU:    resource.MustParse("10000"),
					api.ResourceMemory: resource.MustParse("100"),
				},
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "fooobaz"},
			},
			Spec: api.NodeSpec{
				Capacity: api.ResourceList{
					api.ResourceCPU:    resource.MustParse("100"),
					api.ResourceMemory: resource.MustParse("10000"),
				},
			},
		}, true},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "foo"},
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "fooobaz"},
			},
			Status: api.NodeStatus{
				HostIP: "1.2.3.4",
			},
		}, false},
		{api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "foo"},
			},
			Status: api.NodeStatus{
				HostIP: "1.2.3.4",
			},
		}, api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   "foo",
				Labels: map[string]string{"bar": "fooobaz"},
			},
		}, true},
	}
	for _, test := range tests {
		errs := ValidateMinionUpdate(&test.oldMinion, &test.minion)
		if test.valid && len(errs) > 0 {
			t.Errorf("Unexpected error: %v", errs)
			t.Logf("%#v vs %#v", test.oldMinion.ObjectMeta, test.minion.ObjectMeta)
		}
		if !test.valid && len(errs) == 0 {
			t.Errorf("Unexpected non-error")
		}
	}
}
