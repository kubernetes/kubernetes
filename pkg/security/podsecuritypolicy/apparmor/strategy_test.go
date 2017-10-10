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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/security/apparmor"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/interfaces"
	"k8s.io/kubernetes/pkg/util/maps"
)

const (
	containerName = "test-c"
)

var (
	// pod annotations
	withoutAppArmor = map[string]string{"foo": "bar"}
	withDefault     = map[string]string{
		"foo": "bar",
		apparmor.ContainerAnnotationKeyPrefix + containerName: apparmor.ProfileRuntimeDefault,
	}
	withLocal = map[string]string{
		"foo": "bar",
		apparmor.ContainerAnnotationKeyPrefix + containerName: apparmor.ProfileNamePrefix + "foo",
	}
	withDisallowed = map[string]string{
		"foo": "bar",
		apparmor.ContainerAnnotationKeyPrefix + containerName: apparmor.ProfileNamePrefix + "bad",
	}

	// psp annotations
	noAppArmor               = map[string]string{"foo": "bar"}
	unconstrainedWithDefault = map[string]string{
		apparmor.DefaultProfileAnnotationKey: apparmor.ProfileRuntimeDefault,
	}
	constrained = map[string]string{
		apparmor.AllowedProfilesAnnotationKey: apparmor.ProfileRuntimeDefault + "," +
			apparmor.ProfileNamePrefix + "foo",
	}
	constrainedWithDefault = map[string]string{
		apparmor.DefaultProfileAnnotationKey: apparmor.ProfileRuntimeDefault,
		apparmor.AllowedProfilesAnnotationKey: apparmor.ProfileRuntimeDefault + "," +
			apparmor.ProfileNamePrefix + "foo",
	}

	container = api.Container{
		Name:  containerName,
		Image: "busybox",
	}
)

func TestGenerate(t *testing.T) {
	type testcase struct {
		pspAnnotations map[string]string
		podAnnotations map[string]string
		expected       map[string]string
	}
	tests := []testcase{{
		pspAnnotations: noAppArmor,
		podAnnotations: withoutAppArmor,
		expected:       withoutAppArmor,
	}, {
		pspAnnotations: unconstrainedWithDefault,
		podAnnotations: withoutAppArmor,
		expected:       withDefault,
	}, {
		pspAnnotations: constrained,
		podAnnotations: withoutAppArmor,
		expected:       withoutAppArmor,
	}, {
		pspAnnotations: constrainedWithDefault,
		podAnnotations: withoutAppArmor,
		expected:       withDefault,
	}}

	// Add unchanging permutations.
	for _, podAnnotations := range []map[string]string{withDefault, withLocal} {
		for _, pspAnnotations := range []map[string]string{noAppArmor, unconstrainedWithDefault, constrained, constrainedWithDefault} {
			tests = append(tests, testcase{
				pspAnnotations: pspAnnotations,
				podAnnotations: podAnnotations,
				expected:       podAnnotations,
			})
		}
	}

	for i, test := range tests {
		s := NewStrategy(test.pspAnnotations)
		msgAndArgs := []interface{}{"testcase[%d]: %s", i, spew.Sdump(test)}
		pod := &api.Pod{
			ObjectMeta: metav1.ObjectMeta{Annotations: maps.CopySS(test.podAnnotations)},
			Spec: api.PodSpec{
				Containers: []api.Container{*container.DeepCopy()},
			},
		}
		err := s.DefaultContainer(pod, &pod.Spec.Containers[0])
		assert.NoError(t, err, msgAndArgs...)
		assert.Equal(t, test.expected, pod.Annotations, msgAndArgs...)
	}
}

func TestValidate(t *testing.T) {
	type testcase struct {
		pspAnnotations map[string]string
		podAnnotations map[string]string
		expectResult   interfaces.Result
	}
	tests := []testcase{
		{pspAnnotations: noAppArmor, podAnnotations: withoutAppArmor, expectResult: interfaces.Allowed},
		{pspAnnotations: noAppArmor, podAnnotations: withDefault, expectResult: interfaces.Allowed},
		{pspAnnotations: noAppArmor, podAnnotations: withLocal, expectResult: interfaces.Allowed},
		{pspAnnotations: noAppArmor, podAnnotations: withDisallowed, expectResult: interfaces.Allowed},

		{pspAnnotations: unconstrainedWithDefault, podAnnotations: withoutAppArmor, expectResult: interfaces.RequiresDefaulting},
		{pspAnnotations: unconstrainedWithDefault, podAnnotations: withDefault, expectResult: interfaces.Allowed},
		{pspAnnotations: unconstrainedWithDefault, podAnnotations: withLocal, expectResult: interfaces.Allowed},
		{pspAnnotations: unconstrainedWithDefault, podAnnotations: withDisallowed, expectResult: interfaces.Allowed},

		{pspAnnotations: constrained, podAnnotations: withoutAppArmor, expectResult: interfaces.Forbidden},
		{pspAnnotations: constrained, podAnnotations: withDefault, expectResult: interfaces.Allowed},
		{pspAnnotations: constrained, podAnnotations: withLocal, expectResult: interfaces.Allowed},
		{pspAnnotations: constrained, podAnnotations: withDisallowed, expectResult: interfaces.Forbidden},

		{pspAnnotations: constrainedWithDefault, podAnnotations: withoutAppArmor, expectResult: interfaces.RequiresDefaulting},
		{pspAnnotations: constrainedWithDefault, podAnnotations: withDefault, expectResult: interfaces.Allowed},
		{pspAnnotations: constrainedWithDefault, podAnnotations: withLocal, expectResult: interfaces.Allowed},
		{pspAnnotations: constrainedWithDefault, podAnnotations: withDisallowed, expectResult: interfaces.Forbidden},
	}

	for i, test := range tests {
		s := NewStrategy(test.pspAnnotations)
		pod, container := makeTestPod(test.podAnnotations)
		result, _ := s.ValidateContainer(pod, container, container.SecurityContext)
		if result.Result != test.expectResult {
			t.Logf("%d: %s\n%v\n%v", i, spew.Sdump(test), pod.Annotations, result.Result)
			t.Errorf("%d: expected %v, got %v", i, test.expectResult, result.Result)
		}
	}
}

func makeTestPod(annotations map[string]string) (*api.Pod, *api.Container) {
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test-pod",
			Annotations: maps.CopySS(annotations),
		},
		Spec: api.PodSpec{
			Containers: []api.Container{container},
		},
	}, &container
}
