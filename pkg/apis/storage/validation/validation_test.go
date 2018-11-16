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
	"fmt"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/features"
)

var (
	deleteReclaimPolicy = api.PersistentVolumeReclaimDelete
	immediateMode1      = storage.VolumeBindingImmediate
	immediateMode2      = storage.VolumeBindingImmediate
	waitingMode         = storage.VolumeBindingWaitForFirstConsumer
	invalidMode         = storage.VolumeBindingMode("foo")
)

func TestValidateStorageClass(t *testing.T) {
	deleteReclaimPolicy := api.PersistentVolumeReclaimPolicy("Delete")
	retainReclaimPolicy := api.PersistentVolumeReclaimPolicy("Retain")
	recycleReclaimPolicy := api.PersistentVolumeReclaimPolicy("Recycle")
	successCases := []storage.StorageClass{
		{
			// empty parameters
			ObjectMeta:        metav1.ObjectMeta{Name: "foo"},
			Provisioner:       "kubernetes.io/foo-provisioner",
			Parameters:        map[string]string{},
			ReclaimPolicy:     &deleteReclaimPolicy,
			VolumeBindingMode: &immediateMode1,
		},
		{
			// nil parameters
			ObjectMeta:        metav1.ObjectMeta{Name: "foo"},
			Provisioner:       "kubernetes.io/foo-provisioner",
			ReclaimPolicy:     &deleteReclaimPolicy,
			VolumeBindingMode: &immediateMode1,
		},
		{
			// some parameters
			ObjectMeta:  metav1.ObjectMeta{Name: "foo"},
			Provisioner: "kubernetes.io/foo-provisioner",
			Parameters: map[string]string{
				"kubernetes.io/foo-parameter": "free/form/string",
				"foo-parameter":               "free-form-string",
				"foo-parameter2":              "{\"embedded\": \"json\", \"with\": {\"structures\":\"inside\"}}",
			},
			ReclaimPolicy:     &deleteReclaimPolicy,
			VolumeBindingMode: &immediateMode1,
		},
		{
			// retain reclaimPolicy
			ObjectMeta:        metav1.ObjectMeta{Name: "foo"},
			Provisioner:       "kubernetes.io/foo-provisioner",
			ReclaimPolicy:     &retainReclaimPolicy,
			VolumeBindingMode: &immediateMode1,
		},
	}

	// Success cases are expected to pass validation.
	for k, v := range successCases {
		if errs := ValidateStorageClass(&v); len(errs) != 0 {
			t.Errorf("Expected success for %d, got %v", k, errs)
		}
	}

	// generate a map longer than maxProvisionerParameterSize
	longParameters := make(map[string]string)
	totalSize := 0
	for totalSize < maxProvisionerParameterSize {
		k := fmt.Sprintf("param/%d", totalSize)
		v := fmt.Sprintf("value-%d", totalSize)
		longParameters[k] = v
		totalSize = totalSize + len(k) + len(v)
	}

	errorCases := map[string]storage.StorageClass{
		"namespace is present": {
			ObjectMeta:    metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
			Provisioner:   "kubernetes.io/foo-provisioner",
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		"invalid provisioner": {
			ObjectMeta:    metav1.ObjectMeta{Name: "foo"},
			Provisioner:   "kubernetes.io/invalid/provisioner",
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		"invalid empty parameter name": {
			ObjectMeta:  metav1.ObjectMeta{Name: "foo"},
			Provisioner: "kubernetes.io/foo",
			Parameters: map[string]string{
				"": "value",
			},
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		"provisioner: Required value": {
			ObjectMeta:    metav1.ObjectMeta{Name: "foo"},
			Provisioner:   "",
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		"too long parameters": {
			ObjectMeta:    metav1.ObjectMeta{Name: "foo"},
			Provisioner:   "kubernetes.io/foo",
			Parameters:    longParameters,
			ReclaimPolicy: &deleteReclaimPolicy,
		},
		"invalid reclaimpolicy": {
			ObjectMeta:    metav1.ObjectMeta{Name: "foo"},
			Provisioner:   "kubernetes.io/foo",
			ReclaimPolicy: &recycleReclaimPolicy,
		},
	}

	// Error cases are not expected to pass validation.
	for testName, storageClass := range errorCases {
		if errs := ValidateStorageClass(&storageClass); len(errs) == 0 {
			t.Errorf("Expected failure for test: %s", testName)
		}
	}
}

func TestAlphaExpandPersistentVolumesFeatureValidation(t *testing.T) {
	deleteReclaimPolicy := api.PersistentVolumeReclaimPolicy("Delete")
	falseVar := false
	testSC := &storage.StorageClass{
		// empty parameters
		ObjectMeta:           metav1.ObjectMeta{Name: "foo"},
		Provisioner:          "kubernetes.io/foo-provisioner",
		Parameters:           map[string]string{},
		ReclaimPolicy:        &deleteReclaimPolicy,
		AllowVolumeExpansion: &falseVar,
		VolumeBindingMode:    &immediateMode1,
	}

	// Enable feature ExpandPersistentVolumes
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExpandPersistentVolumes, true)()
	if errs := ValidateStorageClass(testSC); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
	// Disable feature ExpandPersistentVolumes
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ExpandPersistentVolumes, false)()
	if errs := ValidateStorageClass(testSC); len(errs) == 0 {
		t.Errorf("expected failure, but got no error")
	}

}

func TestVolumeAttachmentValidation(t *testing.T) {
	volumeName := "pv-name"
	empty := ""
	successCases := []storage.VolumeAttachment{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
				NodeName: "mynode",
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo-with-status"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
				NodeName: "mynode",
			},
			Status: storage.VolumeAttachmentStatus{
				Attached: true,
				AttachmentMetadata: map[string]string{
					"foo": "bar",
				},
				AttachError: &storage.VolumeError{
					Time:    metav1.Time{},
					Message: "hello world",
				},
				DetachError: &storage.VolumeError{
					Time:    metav1.Time{},
					Message: "hello world",
				},
			},
		},
	}

	for _, volumeAttachment := range successCases {
		if errs := ValidateVolumeAttachment(&volumeAttachment); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	errorCases := []storage.VolumeAttachment{
		{
			// Empty attacher name
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "",
				NodeName: "mynode",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
			},
		},
		{
			// Empty node name
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				NodeName: "",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
			},
		},
		{
			// No volume name
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				NodeName: "node",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: nil,
				},
			},
		},
		{
			// Empty volume name
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				NodeName: "node",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &empty,
				},
			},
		},
		{
			// Too long error message
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				NodeName: "node",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
			},
			Status: storage.VolumeAttachmentStatus{
				Attached: true,
				AttachmentMetadata: map[string]string{
					"foo": "bar",
				},
				AttachError: &storage.VolumeError{
					Time:    metav1.Time{},
					Message: "hello world",
				},
				DetachError: &storage.VolumeError{
					Time:    metav1.Time{},
					Message: strings.Repeat("a", maxVolumeErrorMessageSize+1),
				},
			},
		},
		{
			// Too long metadata
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				NodeName: "node",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
			},
			Status: storage.VolumeAttachmentStatus{
				Attached: true,
				AttachmentMetadata: map[string]string{
					"foo": strings.Repeat("a", maxAttachedVolumeMetadataSize),
				},
				AttachError: &storage.VolumeError{
					Time:    metav1.Time{},
					Message: "hello world",
				},
				DetachError: &storage.VolumeError{
					Time:    metav1.Time{},
					Message: "hello world",
				},
			},
		},
	}

	for _, volumeAttachment := range errorCases {
		if errs := ValidateVolumeAttachment(&volumeAttachment); len(errs) == 0 {
			t.Errorf("Expected failure for test: %v", volumeAttachment)
		}
	}
}

func TestVolumeAttachmentUpdateValidation(t *testing.T) {
	volumeName := "foo"
	newVolumeName := "bar"

	old := storage.VolumeAttachment{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: storage.VolumeAttachmentSpec{
			Attacher: "myattacher",
			Source: storage.VolumeAttachmentSource{
				PersistentVolumeName: &volumeName,
			},
			NodeName: "mynode",
		},
	}
	successCases := []storage.VolumeAttachment{
		{
			// no change
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
				NodeName: "mynode",
			},
		},
		{
			// modify status
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
				NodeName: "mynode",
			},
			Status: storage.VolumeAttachmentStatus{
				Attached: true,
				AttachmentMetadata: map[string]string{
					"foo": "bar",
				},
				AttachError: &storage.VolumeError{
					Time:    metav1.Time{},
					Message: "hello world",
				},
				DetachError: &storage.VolumeError{
					Time:    metav1.Time{},
					Message: "hello world",
				},
			},
		},
	}

	for _, volumeAttachment := range successCases {
		if errs := ValidateVolumeAttachmentUpdate(&volumeAttachment, &old); len(errs) != 0 {
			t.Errorf("expected success: %+v", errs)
		}
	}

	errorCases := []storage.VolumeAttachment{
		{
			// change attacher
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "another-attacher",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
				NodeName: "mynode",
			},
		},
		{
			// change volume
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &newVolumeName,
				},
				NodeName: "mynode",
			},
		},
		{
			// change node
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
				NodeName: "anothernode",
			},
		},
		{
			// add invalid status
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
				NodeName: "mynode",
			},
			Status: storage.VolumeAttachmentStatus{
				Attached: true,
				AttachmentMetadata: map[string]string{
					"foo": "bar",
				},
				AttachError: &storage.VolumeError{
					Time:    metav1.Time{},
					Message: strings.Repeat("a", maxAttachedVolumeMetadataSize),
				},
				DetachError: &storage.VolumeError{
					Time:    metav1.Time{},
					Message: "hello world",
				},
			},
		},
	}

	for _, volumeAttachment := range errorCases {
		if errs := ValidateVolumeAttachmentUpdate(&volumeAttachment, &old); len(errs) == 0 {
			t.Errorf("Expected failure for test: %+v", volumeAttachment)
		}
	}
}

func TestVolumeAttachmentValidationV1(t *testing.T) {
	volumeName := "pv-name"
	invalidVolumeName := "-invalid-@#$%^&*()-"
	successCases := []storage.VolumeAttachment{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
				NodeName: "mynode",
			},
		},
	}

	for _, volumeAttachment := range successCases {
		if errs := ValidateVolumeAttachmentV1(&volumeAttachment); len(errs) != 0 {
			t.Errorf("expected success: %+v", errs)
		}
	}

	errorCases := []storage.VolumeAttachment{
		{
			// Invalid attacher name
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "invalid-@#$%^&*()",
				NodeName: "mynode",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &volumeName,
				},
			},
		},
		{
			// Invalid PV name
			ObjectMeta: metav1.ObjectMeta{Name: "foo"},
			Spec: storage.VolumeAttachmentSpec{
				Attacher: "myattacher",
				NodeName: "mynode",
				Source: storage.VolumeAttachmentSource{
					PersistentVolumeName: &invalidVolumeName,
				},
			},
		},
	}

	for _, volumeAttachment := range errorCases {
		if errs := ValidateVolumeAttachmentV1(&volumeAttachment); len(errs) == 0 {
			t.Errorf("Expected failure for test: %+v", volumeAttachment)
		}
	}
}

func makeClass(mode *storage.VolumeBindingMode, topologies []api.TopologySelectorTerm) *storage.StorageClass {
	return &storage.StorageClass{
		ObjectMeta:        metav1.ObjectMeta{Name: "foo", ResourceVersion: "foo"},
		Provisioner:       "kubernetes.io/foo-provisioner",
		ReclaimPolicy:     &deleteReclaimPolicy,
		VolumeBindingMode: mode,
		AllowedTopologies: topologies,
	}
}

// TODO: Remove these tests once feature gate is not required
func TestValidateVolumeBindingModeAlphaDisabled(t *testing.T) {
	errorCases := map[string]*storage.StorageClass{
		"immediate mode": makeClass(&immediateMode1, nil),
		"waiting mode":   makeClass(&waitingMode, nil),
		"invalid mode":   makeClass(&invalidMode, nil),
	}

	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeScheduling, false)()
	for testName, storageClass := range errorCases {
		if errs := ValidateStorageClass(storageClass); len(errs) == 0 {
			t.Errorf("Expected failure for test: %v", testName)
		}
	}
}

type bindingTest struct {
	class         *storage.StorageClass
	shouldSucceed bool
}

func TestValidateVolumeBindingMode(t *testing.T) {
	cases := map[string]bindingTest{
		"no mode": {
			class:         makeClass(nil, nil),
			shouldSucceed: false,
		},
		"immediate mode": {
			class:         makeClass(&immediateMode1, nil),
			shouldSucceed: true,
		},
		"waiting mode": {
			class:         makeClass(&waitingMode, nil),
			shouldSucceed: true,
		},
		"invalid mode": {
			class:         makeClass(&invalidMode, nil),
			shouldSucceed: false,
		},
	}

	// TODO: remove when feature gate not required
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeScheduling, true)()
	for testName, testCase := range cases {
		errs := ValidateStorageClass(testCase.class)
		if testCase.shouldSucceed && len(errs) != 0 {
			t.Errorf("Expected success for test %q, got %v", testName, errs)
		}
		if !testCase.shouldSucceed && len(errs) == 0 {
			t.Errorf("Expected failure for test %q, got success", testName)
		}
	}
}

type updateTest struct {
	oldClass      *storage.StorageClass
	newClass      *storage.StorageClass
	shouldSucceed bool
}

func TestValidateUpdateVolumeBindingMode(t *testing.T) {
	noBinding := makeClass(nil, nil)
	immediateBinding1 := makeClass(&immediateMode1, nil)
	immediateBinding2 := makeClass(&immediateMode2, nil)
	waitBinding := makeClass(&waitingMode, nil)

	cases := map[string]updateTest{
		"old and new no mode": {
			oldClass:      noBinding,
			newClass:      noBinding,
			shouldSucceed: true,
		},
		"old and new same mode ptr": {
			oldClass:      immediateBinding1,
			newClass:      immediateBinding1,
			shouldSucceed: true,
		},
		"old and new same mode value": {
			oldClass:      immediateBinding1,
			newClass:      immediateBinding2,
			shouldSucceed: true,
		},
		"old no mode, new mode": {
			oldClass:      noBinding,
			newClass:      waitBinding,
			shouldSucceed: false,
		},
		"old mode, new no mode": {
			oldClass:      waitBinding,
			newClass:      noBinding,
			shouldSucceed: false,
		},
		"old and new different modes": {
			oldClass:      waitBinding,
			newClass:      immediateBinding1,
			shouldSucceed: false,
		},
	}

	// TODO: remove when feature gate not required
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeScheduling, true)()
	for testName, testCase := range cases {
		errs := ValidateStorageClassUpdate(testCase.newClass, testCase.oldClass)
		if testCase.shouldSucceed && len(errs) != 0 {
			t.Errorf("Expected success for %v, got %v", testName, errs)
		}
		if !testCase.shouldSucceed && len(errs) == 0 {
			t.Errorf("Expected failure for %v, got success", testName)
		}
	}
}

func TestValidateAllowedTopologies(t *testing.T) {

	validTopology := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "failure-domain.beta.kubernetes.io/zone",
					Values: []string{"zone1"},
				},
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1"},
				},
			},
		},
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "failure-domain.beta.kubernetes.io/zone",
					Values: []string{"zone2"},
				},
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node2"},
				},
			},
		},
	}

	topologyInvalidKey := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "/invalidkey",
					Values: []string{"zone1"},
				},
			},
		},
	}

	topologyLackOfValues := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{},
				},
			},
		},
	}

	topologyDupValues := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1", "node1"},
				},
			},
		},
	}

	topologyMultiValues := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1", "node2"},
				},
			},
		},
	}

	topologyEmptyMatchLabelExpressions := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: nil,
		},
	}

	topologyDupKeys := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1"},
				},
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node2"},
				},
			},
		},
	}

	topologyMultiTerm := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1"},
				},
			},
		},
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node2"},
				},
			},
		},
	}

	topologyDupTermsIdentical := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "failure-domain.beta.kubernetes.io/zone",
					Values: []string{"zone1"},
				},
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1"},
				},
			},
		},
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "failure-domain.beta.kubernetes.io/zone",
					Values: []string{"zone1"},
				},
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1"},
				},
			},
		},
	}

	topologyExprsOneSameOneDiff := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "failure-domain.beta.kubernetes.io/zone",
					Values: []string{"zone1"},
				},
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1"},
				},
			},
		},
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "failure-domain.beta.kubernetes.io/zone",
					Values: []string{"zone1"},
				},
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node2"},
				},
			},
		},
	}

	topologyValuesOneSameOneDiff := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1", "node2"},
				},
			},
		},
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1", "node3"},
				},
			},
		},
	}

	topologyDupTermsDiffExprOrder := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1"},
				},
				{
					Key:    "failure-domain.beta.kubernetes.io/zone",
					Values: []string{"zone1"},
				},
			},
		},
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "failure-domain.beta.kubernetes.io/zone",
					Values: []string{"zone1"},
				},
				{
					Key:    "kubernetes.io/hostname",
					Values: []string{"node1"},
				},
			},
		},
	}

	topologyDupTermsDiffValueOrder := []api.TopologySelectorTerm{
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "failure-domain.beta.kubernetes.io/zone",
					Values: []string{"zone1", "zone2"},
				},
			},
		},
		{
			MatchLabelExpressions: []api.TopologySelectorLabelRequirement{
				{
					Key:    "failure-domain.beta.kubernetes.io/zone",
					Values: []string{"zone2", "zone1"},
				},
			},
		},
	}

	cases := map[string]bindingTest{
		"no topology": {
			class:         makeClass(&waitingMode, nil),
			shouldSucceed: true,
		},
		"valid topology": {
			class:         makeClass(&waitingMode, validTopology),
			shouldSucceed: true,
		},
		"topology invalid key": {
			class:         makeClass(&waitingMode, topologyInvalidKey),
			shouldSucceed: false,
		},
		"topology lack of values": {
			class:         makeClass(&waitingMode, topologyLackOfValues),
			shouldSucceed: false,
		},
		"duplicate TopologySelectorRequirement values": {
			class:         makeClass(&waitingMode, topologyDupValues),
			shouldSucceed: false,
		},
		"multiple TopologySelectorRequirement values": {
			class:         makeClass(&waitingMode, topologyMultiValues),
			shouldSucceed: true,
		},
		"empty MatchLabelExpressions": {
			class:         makeClass(&waitingMode, topologyEmptyMatchLabelExpressions),
			shouldSucceed: false,
		},
		"duplicate MatchLabelExpression keys": {
			class:         makeClass(&waitingMode, topologyDupKeys),
			shouldSucceed: false,
		},
		"duplicate MatchLabelExpression keys but across separate terms": {
			class:         makeClass(&waitingMode, topologyMultiTerm),
			shouldSucceed: true,
		},
		"duplicate AllowedTopologies terms - identical": {
			class:         makeClass(&waitingMode, topologyDupTermsIdentical),
			shouldSucceed: false,
		},
		"two AllowedTopologies terms, with a pair of the same MatchLabelExpressions and a pair of different ones": {
			class:         makeClass(&waitingMode, topologyExprsOneSameOneDiff),
			shouldSucceed: true,
		},
		"two AllowedTopologies terms, with a pair of the same Values and a pair of different ones": {
			class:         makeClass(&waitingMode, topologyValuesOneSameOneDiff),
			shouldSucceed: true,
		},
		"duplicate AllowedTopologies terms - different MatchLabelExpressions order": {
			class:         makeClass(&waitingMode, topologyDupTermsDiffExprOrder),
			shouldSucceed: false,
		},
		"duplicate AllowedTopologies terms - different TopologySelectorRequirement values order": {
			class:         makeClass(&waitingMode, topologyDupTermsDiffValueOrder),
			shouldSucceed: false,
		},
	}

	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeScheduling, true)()
	for testName, testCase := range cases {
		errs := ValidateStorageClass(testCase.class)
		if testCase.shouldSucceed && len(errs) != 0 {
			t.Errorf("Expected success for test %q, got %v", testName, errs)
		}
		if !testCase.shouldSucceed && len(errs) == 0 {
			t.Errorf("Expected failure for test %q, got success", testName)
		}
	}

	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeScheduling, false)()
	for testName, testCase := range cases {
		errs := ValidateStorageClass(testCase.class)
		if len(errs) == 0 && testCase.class.AllowedTopologies != nil {
			t.Errorf("Expected failure for test %q, got success", testName)
		}
	}
}
