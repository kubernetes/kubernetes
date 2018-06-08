/*
Copyright 2018 The Kubernetes Authors.

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

	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestRatchetingPod(t *testing.T) {
	testcases := []struct {
		name           string
		invalidPodSpec *core.PodSpec
		validPodSpec   *core.PodSpec
		err            string
	}{
		{
			name:           "flexvolume",
			invalidPodSpec: &core.PodSpec{Volumes: []core.Volume{{VolumeSource: core.VolumeSource{FlexVolume: &core.FlexVolumeSource{Driver: "invalid driver"}}}}},
			validPodSpec:   &core.PodSpec{Volumes: []core.Volume{{VolumeSource: core.VolumeSource{FlexVolume: &core.FlexVolumeSource{Driver: "valid/driver"}}}}},
			err:            "flexVolume.driver",
		},
		{
			name:           "emptydir medium",
			invalidPodSpec: &core.PodSpec{Volumes: []core.Volume{{VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{Medium: "invalid medium"}}}}},
			validPodSpec:   &core.PodSpec{Volumes: []core.Volume{{VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{Medium: "Memory"}}}}},
			err:            "emptyDir.medium",
		},
		{
			name:           "duplicate envvar names",
			invalidPodSpec: &core.PodSpec{Containers: []core.Container{{Env: []core.EnvVar{{Name: "a"}, {Name: "a"}}}}},
			validPodSpec:   &core.PodSpec{Containers: []core.Container{{Env: []core.EnvVar{{Name: "a"}, {Name: "b"}}}}},
			err:            "env[1].name",
		},
		{
			name: "duplicate pvc names",
			invalidPodSpec: &core.PodSpec{Volumes: []core.Volume{
				{VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "a"}}},
				{VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "a"}}},
			}},
			validPodSpec: &core.PodSpec{Volumes: []core.Volume{
				{VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "a"}}},
				{VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "b"}}},
			}},
			err: "volumes[1].persistentVolumeClaim.claimName",
		},
		{
			name:           "memory units (fractional)",
			invalidPodSpec: &core.PodSpec{Containers: []core.Container{{Resources: core.ResourceRequirements{Limits: getResourceLimits("1000", "1m")}}}},
			validPodSpec:   &core.PodSpec{Containers: []core.Container{{Resources: core.ResourceRequirements{Limits: getResourceLimits("1000", "4Mi")}}}},
			err:            "resources.limits.memory",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			createValidErrs := apimachineryvalidation.ValidateRatchetingCreate(tc.validPodSpec.DeepCopy(), nil, RatchetingPodSpecValidations)
			if len(createValidErrs) > 0 {
				t.Errorf("unexpected errors: %v", createValidErrs)
			}

			createInvalidErrs := apimachineryvalidation.ValidateRatchetingCreate(tc.invalidPodSpec.DeepCopy(), nil, RatchetingPodSpecValidations)
			if len(createInvalidErrs) == 0 {
				t.Errorf("expected errors, got none")
			} else if !strings.Contains(createInvalidErrs.ToAggregate().Error(), tc.err) {
				t.Errorf("expected '%s', got '%s'", tc.err, createInvalidErrs.ToAggregate().Error())
			}

			// valid -> valid == ok
			validToValidErrs := apimachineryvalidation.ValidateRatchetingUpdate(tc.validPodSpec.DeepCopy(), tc.validPodSpec.DeepCopy(), nil, RatchetingPodSpecValidations)
			if len(validToValidErrs) > 0 {
				t.Errorf("unexpected errors: %v", validToValidErrs)
			}

			// invalid -> invalid == ok
			invalidToInvalidErrs := apimachineryvalidation.ValidateRatchetingUpdate(tc.invalidPodSpec.DeepCopy(), tc.invalidPodSpec.DeepCopy(), nil, RatchetingPodSpecValidations)
			if len(invalidToInvalidErrs) > 0 {
				t.Errorf("unexpected errors: %v", invalidToInvalidErrs)
			}

			// invalid -> valid == ok
			invalidToValidErrs := apimachineryvalidation.ValidateRatchetingUpdate(tc.validPodSpec.DeepCopy(), tc.invalidPodSpec.DeepCopy(), nil, RatchetingPodSpecValidations)
			if len(invalidToValidErrs) > 0 {
				t.Errorf("unexpected errors: %v", invalidToValidErrs)
			}

			// valid -> invalid == fail
			validToInvalidErrs := apimachineryvalidation.ValidateRatchetingUpdate(tc.invalidPodSpec.DeepCopy(), tc.validPodSpec.DeepCopy(), nil, RatchetingPodSpecValidations)
			if len(validToInvalidErrs) == 0 {
				t.Errorf("expected errors, got none")
			} else if !strings.Contains(validToInvalidErrs.ToAggregate().Error(), tc.err) {
				t.Errorf("expected '%s', got '%s'", tc.err, validToInvalidErrs.ToAggregate().Error())
			}
		})
	}
}
