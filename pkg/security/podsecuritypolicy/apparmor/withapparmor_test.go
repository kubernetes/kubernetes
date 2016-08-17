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

package apparmor

import (
	"testing"

	"github.com/davecgh/go-spew/spew"
	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/security/apparmor"
)

const (
	containerName = "test-c"
)

var (
	withoutAppArmor = map[string]string{"foo": "bar"}
	withDefault     = map[string]string{
		"foo": "bar",
		apparmor.ContainerAnnotationKeyPrefix + containerName: apparmor.ProfileRuntimeDefault,
	}
	withLocal = map[string]string{
		"foo": "bar",
		apparmor.ContainerAnnotationKeyPrefix + containerName: apparmor.ProfileNamePrefix + "foo",
	}

	disallowed      = []string{}
	allowedDefault  = []string{apparmor.ProfileRuntimeDefault}
	allowedMultiple = []string{apparmor.ProfileRuntimeDefault, apparmor.ProfileNamePrefix + "foo"}
)

func TestGenerate(t *testing.T) {
	type testcase struct {
		allowed     []string
		annotations map[string]string
		expected    map[string]string
	}
	tests := []testcase{{
		allowed:     allowedDefault,
		annotations: withoutAppArmor,
		expected:    withDefault,
	}, {
		allowed:     allowedMultiple,
		annotations: withoutAppArmor,
		expected:    withDefault,
	}, {
		allowed:     disallowed,
		annotations: withoutAppArmor,
		expected:    withoutAppArmor,
	}}

	// Add unchanging permutations.
	for _, allowed := range [][]string{disallowed, allowedDefault, allowedMultiple} {
		for _, annotations := range []map[string]string{withDefault, withLocal} {
			tests = append(tests, testcase{
				allowed:     allowed,
				annotations: annotations,
				expected:    annotations,
			})
		}
	}

	for i, test := range tests {
		s := NewWithAppArmorProfile(test.allowed)
		pod, container := makeTestPod(test.annotations)
		msgAndArgs := []interface{}{"testcase[%d]: %s", i, spew.Sdump(test)}
		annotations, err := s.Generate(pod.Annotations, container)
		assert.NoError(t, err, msgAndArgs...)
		assert.Equal(t, test.expected, annotations, msgAndArgs...)
	}
}

func TestValidate(t *testing.T) {
	type testcase struct {
		allowed     []string
		annotations map[string]string
		expectErr   bool
	}
	tests := []testcase{{
		allowed:     disallowed,
		annotations: withoutAppArmor,
		expectErr:   false,
	}, {
		allowed:     disallowed,
		annotations: withDefault,
		expectErr:   true,
	}, {
		allowed:     disallowed,
		annotations: withLocal,
		expectErr:   true,
	}, {
		allowed:     allowedDefault,
		annotations: withoutAppArmor,
		expectErr:   true,
	}, {
		allowed:     allowedDefault,
		annotations: withDefault,
		expectErr:   false,
	}, {
		allowed:     allowedDefault,
		annotations: withLocal,
		expectErr:   true,
	}, {
		allowed:     allowedMultiple,
		annotations: withoutAppArmor,
		expectErr:   true,
	}, {
		allowed:     allowedMultiple,
		annotations: withDefault,
		expectErr:   false,
	}, {
		allowed:     allowedMultiple,
		annotations: withLocal,
		expectErr:   false,
	}}

	for i, test := range tests {
		s := NewWithAppArmorProfile(test.allowed)
		pod, container := makeTestPod(test.annotations)
		msgAndArgs := []interface{}{"testcase[%d]: %s", i, spew.Sdump(test)}
		errs := s.Validate(pod, container)
		if test.expectErr {
			assert.Len(t, errs, 1, msgAndArgs...)
		} else {
			assert.Len(t, errs, 0, msgAndArgs...)
		}
	}
}

func makeTestPod(annotations map[string]string) (*api.Pod, *api.Container) {
	copy, _ := api.Scheme.DeepCopy(annotations)
	container := api.Container{
		Name:  containerName,
		Image: "busybox",
	}
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:        "test-pod",
			Annotations: copy.(map[string]string),
		},
		Spec: api.PodSpec{
			Containers: []api.Container{container},
		},
	}, &container
}
