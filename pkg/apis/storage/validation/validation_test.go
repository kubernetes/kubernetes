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

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/features"
	utilpointer "k8s.io/utils/pointer"
)

var (
	deleteReclaimPolicy = api.PersistentVolumeReclaimDelete
	immediateMode1      = storage.VolumeBindingImmediate
	immediateMode2      = storage.VolumeBindingImmediate
	waitingMode         = storage.VolumeBindingWaitForFirstConsumer
	invalidMode         = storage.VolumeBindingMode("foo")
	inlineSpec          = api.PersistentVolumeSpec{
		AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
		PersistentVolumeSource: api.PersistentVolumeSource{
			CSI: &api.CSIPersistentVolumeSource{
				Driver:       "com.test.foo",
				VolumeHandle: "foobar",
			},
		},
	}
	longerIDValidateOption = CSINodeValidationOptions{
		AllowLongNodeID: true,
	}
	shorterIDValidationOption = CSINodeValidationOptions{
		AllowLongNodeID: false,
	}
)

func makeValidStorageClass() *storage.StorageClass {

	deleteReclaimPolicy := api.PersistentVolumeReclaimPolicy("Delete")
	return &storage.StorageClass{
		ObjectMeta:    metav1.ObjectMeta{Name: "foo"},
		Provisioner:   "kubernetes.io/foo-provisioner",
		ReclaimPolicy: &deleteReclaimPolicy,
	}
}

func makeStorageClassCustom(tweaks ...func(storageClass *storage.StorageClass)) *storage.StorageClass {
	storageClass := makeValidStorageClass()
	for _, fn := range tweaks {
		fn(storageClass)
	}
	return storageClass
}

func TestValidateStorageClass(t *testing.T) {
	retainReclaimPolicy := api.PersistentVolumeReclaimPolicy("Retain")
	recycleReclaimPolicy := api.PersistentVolumeReclaimPolicy("Recycle")
	setEmptyParameters := func(storageClass *storage.StorageClass) {
		storageClass.Parameters = map[string]string{}
	}
	setImmediateVolumeBindingMode := func(storageClass *storage.StorageClass) {
		storageClass.VolumeBindingMode = &immediateMode1
	}
	setMultiParameters := func(storageClass *storage.StorageClass) {
		storageClass.Parameters = map[string]string{
			"kubernetes.io/foo-parameter": "free/form/string",
			"foo-parameter":               "free-form-string",
			"foo-parameter2":              "{\"embedded\": \"json\", \"with\": {\"structures\":\"inside\"}}",
		}
	}
	setRetrainReClaimPolicy := func(storageClass *storage.StorageClass) {
		storageClass.ReclaimPolicy = &retainReclaimPolicy
	}
	successCases := []*storage.StorageClass{
		makeStorageClassCustom(setEmptyParameters, setImmediateVolumeBindingMode),
		makeStorageClassCustom(setImmediateVolumeBindingMode),
		makeStorageClassCustom(setMultiParameters, setImmediateVolumeBindingMode),
		makeStorageClassCustom(setRetrainReClaimPolicy, setImmediateVolumeBindingMode),
	}

	// Success cases are expected to pass validation.
	for k, v := range successCases {
		if errs := ValidateStorageClass(v); len(errs) != 0 {
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
	setNamespaceIsPresent := func(storageClass *storage.StorageClass) {
		storageClass.ObjectMeta = metav1.ObjectMeta{Name: "foo", Namespace: "bar"}
	}
	setInvalidProvisioner := func(storageClass *storage.StorageClass) {
		storageClass.Provisioner = "kubernetes.io/invalid/provisioner"
	}
	setEmptyParameterName := func(storageClass *storage.StorageClass) {
		storageClass.Parameters = map[string]string{"": "value"}
	}
	setEmptyProvisioner := func(storageClass *storage.StorageClass) {
		storageClass.Provisioner = ""
	}
	setTooLongParameters := func(storageClass *storage.StorageClass) {
		storageClass.Parameters = longParameters
	}
	setInvalidReclaimPolicy := func(storageClass *storage.StorageClass) {
		storageClass.ReclaimPolicy = &recycleReclaimPolicy
	}

	errorCases := map[string]*storage.StorageClass{
		"namespace is present":         makeStorageClassCustom(setNamespaceIsPresent),
		"invalid provisioner":          makeStorageClassCustom(setInvalidProvisioner),
		"invalid empty parameter name": makeStorageClassCustom(setEmptyParameterName),
		"provisioner: Required value":  makeStorageClassCustom(setEmptyProvisioner),
		"too long parameters":          makeStorageClassCustom(setTooLongParameters),
		"invalid reclaimpolicy":        makeStorageClassCustom(setInvalidReclaimPolicy),
	}

	// Error cases are not expected to pass validation.
	for testName, storageClass := range errorCases {
		if errs := ValidateStorageClass(storageClass); len(errs) == 0 {
			t.Errorf("Expected failure for test: %s", testName)
		}
	}
}

func makeValidVolumeAttachment() *storage.VolumeAttachment {

	return &storage.VolumeAttachment{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: storage.VolumeAttachmentSpec{
			Attacher: "myattacher",
			NodeName: "mynode",
		},
	}
}

func makeVolumeAttachmentCustom(tweaks ...func(volumeAttachment *storage.VolumeAttachment)) *storage.VolumeAttachment {
	volumeAttachment := makeValidVolumeAttachment()
	for _, fn := range tweaks {
		fn(volumeAttachment)
	}
	return volumeAttachment
}

func TestVolumeAttachmentValidation(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, true)()
	volumeName := "pv-name"
	empty := ""
	setSpecSourceVolumeName := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{
			PersistentVolumeName: &volumeName,
		}
	}
	setMetaNameInlineSpec := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.ObjectMeta = metav1.ObjectMeta{Name: "foo-with-inlinespec"}
	}
	setMetaNameWithStatus := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.ObjectMeta = metav1.ObjectMeta{Name: "foo-with-status"}
	}
	setMetaNameInlineAndStatus := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.ObjectMeta = metav1.ObjectMeta{Name: "foo-with-status"}
	}
	setSpecSourceInline := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{
			InlineVolumeSpec: &inlineSpec,
		}
	}
	setWithStatus := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Status = storage.VolumeAttachmentStatus{
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
		}
	}

	migrationEnabledSuccessCases := []*storage.VolumeAttachment{
		makeVolumeAttachmentCustom(setSpecSourceVolumeName),
		makeVolumeAttachmentCustom(setMetaNameInlineSpec, setSpecSourceInline),
		makeVolumeAttachmentCustom(setMetaNameWithStatus, setSpecSourceVolumeName, setWithStatus),
		makeVolumeAttachmentCustom(setMetaNameInlineAndStatus, setSpecSourceInline, setWithStatus),
	}

	for _, volumeAttachment := range migrationEnabledSuccessCases {
		if errs := ValidateVolumeAttachment(volumeAttachment); len(errs) != 0 {
			t.Errorf("expected success: %v %v", volumeAttachment, errs)
		}
	}
	setEmptyAttacher := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Attacher = ""
	}
	setEmptyNodeName := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.NodeName = ""
	}
	setNilPVName := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{PersistentVolumeName: nil}
	}
	setEmptyPVName := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{PersistentVolumeName: &empty}
	}
	setTooLongErrorMessage := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Status.DetachError = &storage.VolumeError{
			Time: metav1.Time{}, Message: strings.Repeat("a", maxVolumeErrorMessageSize+1)}
	}
	setTooLongMetadata := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Status.AttachmentMetadata = map[string]string{"foo": strings.Repeat("a", maxAttachedVolumeMetadataSize)}
	}
	setWithNoPVNorInlineSpec := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{}
	}
	setWithPVAndInlineSpec := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{
			PersistentVolumeName: &volumeName,
			InlineVolumeSpec:     &inlineSpec,
		}
	}
	setWithInlineWithoutPVSource := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{
			PersistentVolumeName: &volumeName,
			InlineVolumeSpec: &api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
				PersistentVolumeSource: api.PersistentVolumeSource{
					FlexVolume: &api.FlexPersistentVolumeSource{
						Driver: "kubernetes.io/blue",
						FSType: "ext4",
					},
				},
				StorageClassName: "test-storage-class",
			},
		}
	}

	migrationEnabledErrorCases := []*storage.VolumeAttachment{
		makeVolumeAttachmentCustom(setSpecSourceVolumeName, setEmptyAttacher),
		makeVolumeAttachmentCustom(setSpecSourceVolumeName, setEmptyNodeName),
		makeVolumeAttachmentCustom(setNilPVName),
		makeVolumeAttachmentCustom(setEmptyPVName),
		makeVolumeAttachmentCustom(setSpecSourceVolumeName, setWithStatus, setTooLongErrorMessage),
		makeVolumeAttachmentCustom(setSpecSourceVolumeName, setWithStatus, setTooLongMetadata),
		makeVolumeAttachmentCustom(setWithNoPVNorInlineSpec),
		makeVolumeAttachmentCustom(setWithPVAndInlineSpec),
		makeVolumeAttachmentCustom(setWithInlineWithoutPVSource),
	}

	for _, volumeAttachment := range migrationEnabledErrorCases {
		if errs := ValidateVolumeAttachment(volumeAttachment); len(errs) == 0 {
			t.Errorf("expected failure for test: %v", volumeAttachment)
		}
	}

	// validate with CSIMigration disabled
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, false)()

	migrationDisabledSuccessCases := []*storage.VolumeAttachment{

		// PVName specified with migration disabled
		makeVolumeAttachmentCustom(setSpecSourceVolumeName),
		// InlineSpec specified with migration disabled
		makeVolumeAttachmentCustom(setSpecSourceInline),
	}
	for _, volumeAttachment := range migrationDisabledSuccessCases {
		if errs := ValidateVolumeAttachment(volumeAttachment); len(errs) != 0 {
			t.Errorf("expected success: %v %v", volumeAttachment, errs)
		}
	}
}

func TestVolumeAttachmentUpdateValidation(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIMigration, true)()
	volumeName := "foo"
	newVolumeName := "bar"
	setNoChangeVolumeAttachmentSource := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{}
	}

	setWithStatus := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Status = storage.VolumeAttachmentStatus{
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
		}
	}

	old := makeVolumeAttachmentCustom(setNoChangeVolumeAttachmentSource)

	successCases := []*storage.VolumeAttachment{
		makeVolumeAttachmentCustom(setNoChangeVolumeAttachmentSource),
		makeVolumeAttachmentCustom(setNoChangeVolumeAttachmentSource, setWithStatus),
	}

	for _, volumeAttachment := range successCases {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{}
		old.Spec.Source = storage.VolumeAttachmentSource{}
		// test scenarios with PersistentVolumeName set
		volumeAttachment.Spec.Source.PersistentVolumeName = &volumeName
		old.Spec.Source.PersistentVolumeName = &volumeName
		if errs := ValidateVolumeAttachmentUpdate(volumeAttachment, old); len(errs) != 0 {
			t.Errorf("expected success: %+v", errs)
		}

		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{}
		old.Spec.Source = storage.VolumeAttachmentSource{}
		// test scenarios with InlineVolumeSpec set
		volumeAttachment.Spec.Source.InlineVolumeSpec = &inlineSpec
		old.Spec.Source.InlineVolumeSpec = &inlineSpec
		if errs := ValidateVolumeAttachmentUpdate(volumeAttachment, old); len(errs) != 0 {
			t.Errorf("expected success: %+v", errs)
		}
	}

	// reset old's source with volumeName in case it was left with something else by earlier tests
	old.Spec.Source = storage.VolumeAttachmentSource{}
	old.Spec.Source.PersistentVolumeName = &volumeName

	setSpecSourceVolumeName := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{
			PersistentVolumeName: &volumeName,
		}
	}
	setSpecSourceNewVolumeName := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{
			PersistentVolumeName: &newVolumeName,
		}
	}

	setAnotherAttacher := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Attacher = "another-attacher"
	}

	setAnotherNodeName := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.NodeName = "anothernode"
	}

	setSpecSourceInLine := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{
			InlineVolumeSpec: &inlineSpec,
		}
	}
	setStatusAttachError := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Status.AttachError = &storage.VolumeError{
			Time:    metav1.Time{},
			Message: strings.Repeat("a", maxAttachedVolumeMetadataSize),
		}
	}

	errorCases := []*storage.VolumeAttachment{
		makeVolumeAttachmentCustom(setSpecSourceVolumeName, setAnotherAttacher),
		makeVolumeAttachmentCustom(setSpecSourceNewVolumeName),
		makeVolumeAttachmentCustom(setSpecSourceVolumeName, setAnotherNodeName),
		makeVolumeAttachmentCustom(setSpecSourceInLine),
		makeVolumeAttachmentCustom(setSpecSourceVolumeName, setWithStatus, setStatusAttachError),
	}

	for _, volumeAttachment := range errorCases {
		if errs := ValidateVolumeAttachmentUpdate(volumeAttachment, old); len(errs) == 0 {
			t.Errorf("Expected failure for test: %+v", volumeAttachment)
		}
	}
}

func TestVolumeAttachmentValidationV1(t *testing.T) {
	volumeName := "pv-name"
	invalidVolumeName := "-invalid-@#$%^&*()-"
	setSpecSourceVolumeName := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{
			PersistentVolumeName: &volumeName,
		}
	}
	successCases := []*storage.VolumeAttachment{
		makeVolumeAttachmentCustom(setSpecSourceVolumeName),
	}

	for _, volumeAttachment := range successCases {
		if errs := ValidateVolumeAttachmentV1(volumeAttachment); len(errs) != 0 {
			t.Errorf("expected success: %+v", errs)
		}
	}

	setSpecInvalidAttacher := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Attacher = "invalid-@#$%^&*()"
	}

	setSpecInvalidPV := func(volumeAttachment *storage.VolumeAttachment) {
		volumeAttachment.Spec.Source = storage.VolumeAttachmentSource{
			PersistentVolumeName: &invalidVolumeName,
		}
	}

	errorCases := []*storage.VolumeAttachment{
		makeVolumeAttachmentCustom(setSpecSourceVolumeName, setSpecInvalidAttacher),
		makeVolumeAttachmentCustom(setSpecInvalidPV),
	}

	for _, volumeAttachment := range errorCases {
		if errs := ValidateVolumeAttachmentV1(volumeAttachment); len(errs) == 0 {
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

func TestCSINodeValidation(t *testing.T) {
	driverName := "driver-name"
	driverName2 := "1io.kubernetes-storage-2-csi-driver3"
	longName := "my-a-b-c-d-c-f-g-h-i-j-k-l-m-n-o-p-q-r-s-t-u-v-w-x-y-z-ABCDEFGHIJKLMNOPQRSTUVWXYZ-driver" // 88 chars
	nodeID := "nodeA"
	longID := longName + longName // 176 chars
	successCases := []storage.CSINode{
		{
			// driver name: dot only
			ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// driver name: dash only
			ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io-kubernetes-storage-csi-driver",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// driver name: numbers
			ObjectMeta: metav1.ObjectMeta{Name: "foo3"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "1io-kubernetes-storage-2-csi-driver3",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// driver name: dot, dash
			ObjectMeta: metav1.ObjectMeta{Name: "foo4"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage-csi-driver",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// driver name: dot, dash, and numbers
			ObjectMeta: metav1.ObjectMeta{Name: "foo5"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         driverName2,
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// Driver name length 1
			ObjectMeta: metav1.ObjectMeta{Name: "foo2"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "a",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// multiple drivers with different node IDs, topology keys
			ObjectMeta: metav1.ObjectMeta{Name: "foo6"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "driver1",
						NodeID:       "node1",
						TopologyKeys: []string{"key1", "key2"},
					},
					{
						Name:         "driverB",
						NodeID:       "nodeA",
						TopologyKeys: []string{"keyA", "keyB"},
					},
				},
			},
		},
		{
			// multiple drivers with same node IDs, topology keys
			ObjectMeta: metav1.ObjectMeta{Name: "foo7"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "driver1",
						NodeID:       "node1",
						TopologyKeys: []string{"key1"},
					},
					{
						Name:         "driver2",
						NodeID:       "node1",
						TopologyKeys: []string{"key1"},
					},
				},
			},
		},
		{
			// Volume limits being zero
			ObjectMeta: metav1.ObjectMeta{Name: "foo11"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(0)},
					},
				},
			},
		},
		{
			// Volume limits with positive number
			ObjectMeta: metav1.ObjectMeta{Name: "foo11"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(1)},
					},
				},
			},
		},
		{
			// topology key names with -, _, and dot .
			ObjectMeta: metav1.ObjectMeta{Name: "foo8"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "driver1",
						NodeID:       "node1",
						TopologyKeys: []string{"zone_1", "zone.2"},
					},
					{
						Name:         "driver2",
						NodeID:       "node1",
						TopologyKeys: []string{"zone-3", "zone.4"},
					},
				},
			},
		},
		{
			// topology prefix with - and dot.
			ObjectMeta: metav1.ObjectMeta{Name: "foo9"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "driver1",
						NodeID:       "node1",
						TopologyKeys: []string{"company-com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// No topology keys
			ObjectMeta: metav1.ObjectMeta{Name: "foo10"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:   driverName,
						NodeID: nodeID,
					},
				},
			},
		},
	}

	for _, csiNode := range successCases {
		if errs := ValidateCSINode(&csiNode, shorterIDValidationOption); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	nodeIDCase := storage.CSINode{
		// node ID length > 128 but < 192
		ObjectMeta: metav1.ObjectMeta{Name: "foo7"},
		Spec: storage.CSINodeSpec{
			Drivers: []storage.CSINodeDriver{
				{
					Name:         driverName,
					NodeID:       longID,
					TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
				},
			},
		},
	}

	if errs := ValidateCSINode(&nodeIDCase, longerIDValidateOption); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := []storage.CSINode{
		{
			// Empty driver name
			ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// Invalid start char in driver name
			ObjectMeta: metav1.ObjectMeta{Name: "foo3"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "_io.kubernetes.storage.csi.driver",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// Invalid end char in driver name
			ObjectMeta: metav1.ObjectMeta{Name: "foo4"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver/",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// Invalid separators in driver name
			ObjectMeta: metav1.ObjectMeta{Name: "foo5"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io/kubernetes/storage/csi~driver",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// driver name: underscore only
			ObjectMeta: metav1.ObjectMeta{Name: "foo6"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io_kubernetes_storage_csi_driver",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// Driver name length > 63
			ObjectMeta: metav1.ObjectMeta{Name: "foo7"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         longName,
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// No driver name
			ObjectMeta: metav1.ObjectMeta{Name: "foo8"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// Empty individual topology key
			ObjectMeta: metav1.ObjectMeta{Name: "foo9"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         driverName,
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", ""},
					},
				},
			},
		},
		{
			// duplicate drivers in driver specs
			ObjectMeta: metav1.ObjectMeta{Name: "foo10"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "driver1",
						NodeID:       "node1",
						TopologyKeys: []string{"key1", "key2"},
					},
					{
						Name:         "driver1",
						NodeID:       "nodeX",
						TopologyKeys: []string{"keyA", "keyB"},
					},
				},
			},
		},
		{
			// single driver with duplicate topology keys in driver specs
			ObjectMeta: metav1.ObjectMeta{Name: "foo11"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "driver1",
						NodeID:       "node1",
						TopologyKeys: []string{"key1", "key1"},
					},
				},
			},
		},
		{
			// multiple drivers with one set of duplicate topology keys in driver specs
			ObjectMeta: metav1.ObjectMeta{Name: "foo12"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "driver1",
						NodeID:       "node1",
						TopologyKeys: []string{"key1"},
					},
					{
						Name:         "driver2",
						NodeID:       "nodeX",
						TopologyKeys: []string{"keyA", "keyA"},
					},
				},
			},
		},
		{
			// Empty NodeID
			ObjectMeta: metav1.ObjectMeta{Name: "foo13"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         driverName,
						NodeID:       "",
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// Volume limits with negative number
			ObjectMeta: metav1.ObjectMeta{Name: "foo11"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(-1)},
					},
				},
			},
		},
		{
			// topology prefix should be lower case
			ObjectMeta: metav1.ObjectMeta{Name: "foo14"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         driverName,
						NodeID:       "node1",
						TopologyKeys: []string{"Company.Com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		nodeIDCase,
	}

	for _, csiNode := range errorCases {
		if errs := ValidateCSINode(&csiNode, shorterIDValidationOption); len(errs) == 0 {
			t.Errorf("Expected failure for test: %v", csiNode)
		}
	}
}

func TestCSINodeUpdateValidation(t *testing.T) {
	//driverName := "driver-name"
	//driverName2 := "1io.kubernetes-storage-2-csi-driver3"
	//longName := "my-a-b-c-d-c-f-g-h-i-j-k-l-m-n-o-p-q-r-s-t-u-v-w-x-y-z-ABCDEFGHIJKLMNOPQRSTUVWXYZ-driver"
	nodeID := "nodeA"

	old := storage.CSINode{
		ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
		Spec: storage.CSINodeSpec{
			Drivers: []storage.CSINodeDriver{
				{
					Name:         "io.kubernetes.storage.csi.driver-1",
					NodeID:       nodeID,
					TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
				},
				{
					Name:         "io.kubernetes.storage.csi.driver-2",
					NodeID:       nodeID,
					TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(20)},
				},
			},
		},
	}

	successCases := []storage.CSINode{
		{
			// no change
			ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver-1",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
					{
						Name:         "io.kubernetes.storage.csi.driver-2",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(20)},
					},
				},
			},
		},
		{
			// remove a driver
			ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver-1",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
		{
			// add a driver
			ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver-1",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
					{
						Name:         "io.kubernetes.storage.csi.driver-2",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(20)},
					},
					{
						Name:         "io.kubernetes.storage.csi.driver-3",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(30)},
					},
				},
			},
		},
		{
			// remove a driver and add a driver
			ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver-1",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
					{
						Name:         "io.kubernetes.storage.csi.new-driver",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(30)},
					},
				},
			},
		},
	}

	for _, csiNode := range successCases {
		if errs := ValidateCSINodeUpdate(&csiNode, &old, shorterIDValidationOption); len(errs) != 0 {
			t.Errorf("expected success: %+v", errs)
		}
	}

	errorCases := []storage.CSINode{
		{
			// invalid change node id
			ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver-1",
						NodeID:       "nodeB",
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
					{
						Name:         "io.kubernetes.storage.csi.driver-2",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(20)},
					},
				},
			},
		},
		{
			// invalid change topology keys
			ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver-1",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
					{
						Name:         "io.kubernetes.storage.csi.driver-2",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(20)},
					},
				},
			},
		},
		{
			// invalid change trying to set a previously unset allocatable
			ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver-1",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(10)},
					},
					{
						Name:         "io.kubernetes.storage.csi.driver-2",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(20)},
					},
				},
			},
		},
		{
			// invalid change trying to update allocatable with a different volume limit
			ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver-1",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
					{
						Name:         "io.kubernetes.storage.csi.driver-2",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: utilpointer.Int32Ptr(21)},
					},
				},
			},
		},
		{
			// invalid change trying to update allocatable with an empty volume limit
			ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver-1",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
					{
						Name:         "io.kubernetes.storage.csi.driver-2",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
						Allocatable:  &storage.VolumeNodeResources{Count: nil},
					},
				},
			},
		},
		{
			// invalid change trying to remove allocatable
			ObjectMeta: metav1.ObjectMeta{Name: "foo1"},
			Spec: storage.CSINodeSpec{
				Drivers: []storage.CSINodeDriver{
					{
						Name:         "io.kubernetes.storage.csi.driver-1",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
					{
						Name:         "io.kubernetes.storage.csi.driver-2",
						NodeID:       nodeID,
						TopologyKeys: []string{"company.com/zone1", "company.com/zone2"},
					},
				},
			},
		},
	}

	for _, csiNode := range errorCases {
		if errs := ValidateCSINodeUpdate(&csiNode, &old, shorterIDValidationOption); len(errs) == 0 {
			t.Errorf("Expected failure for test: %+v", csiNode)
		}
	}
}

func makeValidCSIDriver() *storage.CSIDriver {
	driverName := "test-driver"
	attachRequired := true
	podInfoOnMount := true
	storageCapacity := true
	return &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{Name: driverName},
		Spec: storage.CSIDriverSpec{
			AttachRequired:  &attachRequired,
			PodInfoOnMount:  &podInfoOnMount,
			StorageCapacity: &storageCapacity,
		},
	}
}

func makeCSIDriverCustom(tweaks ...func(csiDriver *storage.CSIDriver)) *storage.CSIDriver {
	csiDriver := makeValidCSIDriver()
	for _, fn := range tweaks {
		fn(csiDriver)
	}
	return csiDriver
}

func TestCSIDriverValidation(t *testing.T) {
	longName := "my-a-b-c-d-c-f-g-h-i-j-k-l-m-n-o-p-q-r-s-t-u-v-w-x-y-z-ABCDEFGHIJKLMNOPQRSTUVWXYZ-driver"
	invalidName := "-invalid-@#$%^&*()-"
	attachNotRequired := false
	notPodInfoOnMount := false
	notRequiresRepublish := false
	notStorageCapacity := false
	supportedFSGroupPolicy := storage.FileFSGroupPolicy
	invalidFSGroupPolicy := storage.ReadWriteOnceWithFSTypeFSGroupPolicy
	invalidFSGroupPolicy = "invalid-mode"

	setSpecNotRequiresRepublish := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.RequiresRepublish = &notRequiresRepublish

	}
	setNameDotOnly := func(csiDriver *storage.CSIDriver) {
		csiDriver.ObjectMeta = metav1.ObjectMeta{Name: "io.kubernetes.storage.csi.driver"}
	}
	setNameDashOnly := func(csiDriver *storage.CSIDriver) {
		csiDriver.ObjectMeta = metav1.ObjectMeta{Name: "io-kubernetes-storage-csi-driver"}
	}
	setSpecPodInfoOnMountFalse := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.PodInfoOnMount = &notPodInfoOnMount
	}
	setNameNumbers := func(csiDriver *storage.CSIDriver) {
		csiDriver.ObjectMeta = metav1.ObjectMeta{Name: "1csi2driver3"}
	}
	setNameDotAndDash := func(csiDriver *storage.CSIDriver) {
		csiDriver.ObjectMeta = metav1.ObjectMeta{Name: "io.kubernetes.storage.csi-driver"}
	}
	setSpecAttachRequiredFalse := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.AttachRequired = &attachNotRequired
	}
	setSpecVolumeLifeCycleP := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
			storage.VolumeLifecyclePersistent,
		}
	}
	setSpecVolumeLifeCycleE := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
			storage.VolumeLifecycleEphemeral,
		}
	}
	setSpecVolumeLifeCyclePE := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
			storage.VolumeLifecycleEphemeral,
			storage.VolumeLifecyclePersistent,
		}
	}
	setSpecVolumeLifeCycleEPE := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
			storage.VolumeLifecycleEphemeral,
			storage.VolumeLifecyclePersistent,
			storage.VolumeLifecycleEphemeral,
		}
	}
	setSpecFsGroupPolicy := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.FSGroupPolicy = &supportedFSGroupPolicy
	}
	setSpecStorageCapacityFalse := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.StorageCapacity = &notStorageCapacity
	}

	successCases := []*storage.CSIDriver{
		makeCSIDriverCustom(setSpecNotRequiresRepublish),
		makeCSIDriverCustom(setNameDotOnly, setSpecNotRequiresRepublish, setSpecStorageCapacityFalse),
		makeCSIDriverCustom(setNameDashOnly, setSpecPodInfoOnMountFalse, setSpecNotRequiresRepublish),
		makeCSIDriverCustom(setNameNumbers, setSpecNotRequiresRepublish),
		makeCSIDriverCustom(setNameDotAndDash, setSpecNotRequiresRepublish),
		makeCSIDriverCustom(setSpecPodInfoOnMountFalse, setSpecNotRequiresRepublish),
		makeCSIDriverCustom(setSpecAttachRequiredFalse, setSpecPodInfoOnMountFalse, setSpecNotRequiresRepublish),
		makeCSIDriverCustom(setSpecVolumeLifeCycleP, setSpecAttachRequiredFalse, setSpecPodInfoOnMountFalse, setSpecNotRequiresRepublish),
		makeCSIDriverCustom(setSpecVolumeLifeCycleE, setSpecAttachRequiredFalse, setSpecPodInfoOnMountFalse, setSpecNotRequiresRepublish),
		makeCSIDriverCustom(setSpecVolumeLifeCyclePE, setSpecAttachRequiredFalse, setSpecPodInfoOnMountFalse, setSpecNotRequiresRepublish),
		makeCSIDriverCustom(setSpecVolumeLifeCycleEPE, setSpecAttachRequiredFalse, setSpecPodInfoOnMountFalse, setSpecNotRequiresRepublish),
		makeCSIDriverCustom(setSpecFsGroupPolicy, setSpecAttachRequiredFalse, setSpecPodInfoOnMountFalse, setSpecNotRequiresRepublish),
	}

	for _, csiDriver := range successCases {
		if errs := ValidateCSIDriver(csiDriver); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	setNameInvalid := func(csiDriver *storage.CSIDriver) {
		csiDriver.ObjectMeta = metav1.ObjectMeta{Name: invalidName}
	}
	setInvalidLongName := func(csiDriver *storage.CSIDriver) {
		csiDriver.ObjectMeta = metav1.ObjectMeta{Name: longName}
	}

	setSpecAttachRequiredNil := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.AttachRequired = nil
	}

	setSpecPodInfoOnMountNil := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.PodInfoOnMount = nil
	}
	setSpecVolumeLifeCycleModeInvalid := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{"no-such-mode"}
	}
	setSpecInvalidFsGroupPolicy := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.FSGroupPolicy = &invalidFSGroupPolicy

	}
	setSpecStorageCapacityNil := func(csiDriver *storage.CSIDriver) {
		csiDriver.Spec.StorageCapacity = nil
	}

	errorCases := []*storage.CSIDriver{
		makeCSIDriverCustom(setNameInvalid),
		makeCSIDriverCustom(setSpecAttachRequiredFalse, setSpecPodInfoOnMountFalse, setInvalidLongName),
		makeCSIDriverCustom(setSpecAttachRequiredNil),
		makeCSIDriverCustom(setSpecAttachRequiredFalse, setSpecPodInfoOnMountNil),
		makeCSIDriverCustom(setSpecAttachRequiredFalse, setSpecStorageCapacityNil),
		makeCSIDriverCustom(setSpecAttachRequiredFalse, setSpecPodInfoOnMountFalse, setSpecVolumeLifeCycleModeInvalid),
		makeCSIDriverCustom(setSpecAttachRequiredFalse, setSpecPodInfoOnMountFalse, setSpecInvalidFsGroupPolicy),
	}

	for _, csiDriver := range errorCases {
		if errs := ValidateCSIDriver(csiDriver); len(errs) == 0 {
			t.Errorf("Expected failure for test: %v", csiDriver)
		}
	}
}

func TestCSIDriverValidationUpdate(t *testing.T) {
	driverName := "test-driver"
	longName := "my-a-b-c-d-c-f-g-h-i-j-k-l-m-n-o-p-q-r-s-t-u-v-w-x-y-z-ABCDEFGHIJKLMNOPQRSTUVWXYZ-driver"
	invalidName := "-invalid-@#$%^&*()-"
	attachRequired := true
	attachNotRequired := false
	podInfoOnMount := true
	storageCapacity := true
	notPodInfoOnMount := false
	gcp := "gcp"
	requiresRepublish := true
	notRequiresRepublish := false
	notStorageCapacity := false
	resourceVersion := "1"
	old := storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{Name: driverName, ResourceVersion: resourceVersion},
		Spec: storage.CSIDriverSpec{
			AttachRequired:    &attachNotRequired,
			PodInfoOnMount:    &notPodInfoOnMount,
			RequiresRepublish: &notRequiresRepublish,
			VolumeLifecycleModes: []storage.VolumeLifecycleMode{
				storage.VolumeLifecycleEphemeral,
				storage.VolumeLifecyclePersistent,
			},
			StorageCapacity: &storageCapacity,
		},
	}

	successCases := []struct {
		name   string
		modify func(new *storage.CSIDriver)
	}{
		{
			name:   "no change",
			modify: func(new *storage.CSIDriver) {},
		},
		{
			name: "change TokenRequests",
			modify: func(new *storage.CSIDriver) {
				new.Spec.TokenRequests = []storage.TokenRequest{{Audience: gcp}}
			},
		},
		{
			name: "change RequiresRepublish",
			modify: func(new *storage.CSIDriver) {
				new.Spec.RequiresRepublish = &requiresRepublish
			},
		},
	}
	for _, test := range successCases {
		t.Run(test.name, func(t *testing.T) {
			new := old.DeepCopy()
			test.modify(new)
			if errs := ValidateCSIDriverUpdate(new, &old); len(errs) != 0 {
				t.Errorf("Expected success for %+v: %v", new, errs)
			}
		})
	}

	// Each test case changes exactly one field. None of that is valid.
	errorCases := []struct {
		name   string
		modify func(new *storage.CSIDriver)
	}{
		{
			name: "invalid name",
			modify: func(new *storage.CSIDriver) {
				new.Name = invalidName
			},
		},
		{
			name: "long name",
			modify: func(new *storage.CSIDriver) {
				new.Name = longName
			},
		},
		{
			name: "AttachRequired not set",
			modify: func(new *storage.CSIDriver) {
				new.Spec.AttachRequired = nil
			},
		},
		{
			name: "AttachRequired changed",
			modify: func(new *storage.CSIDriver) {
				new.Spec.AttachRequired = &attachRequired
			},
		},
		{
			name: "PodInfoOnMount not set",
			modify: func(new *storage.CSIDriver) {
				new.Spec.PodInfoOnMount = nil
			},
		},
		{
			name: "PodInfoOnMount changed",
			modify: func(new *storage.CSIDriver) {
				new.Spec.PodInfoOnMount = &podInfoOnMount
			},
		},
		{
			name: "invalid volume lifecycle mode",
			modify: func(new *storage.CSIDriver) {
				new.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
					"no-such-mode",
				}
			},
		},
		{
			name: "volume lifecycle modes not set",
			modify: func(new *storage.CSIDriver) {
				new.Spec.VolumeLifecycleModes = nil
			},
		},
		{
			name: "VolumeLifecyclePersistent removed",
			modify: func(new *storage.CSIDriver) {
				new.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
					storage.VolumeLifecycleEphemeral,
				}
			},
		},
		{
			name: "VolumeLifecycleEphemeral removed",
			modify: func(new *storage.CSIDriver) {
				new.Spec.VolumeLifecycleModes = []storage.VolumeLifecycleMode{
					storage.VolumeLifecyclePersistent,
				}
			},
		},
		{
			name: "FSGroupPolicy invalidated",
			modify: func(new *storage.CSIDriver) {
				invalidFSGroupPolicy := storage.ReadWriteOnceWithFSTypeFSGroupPolicy
				invalidFSGroupPolicy = "invalid"
				new.Spec.FSGroupPolicy = &invalidFSGroupPolicy
			},
		},
		{
			name: "FSGroupPolicy changed",
			modify: func(new *storage.CSIDriver) {
				fileFSGroupPolicy := storage.FileFSGroupPolicy
				new.Spec.FSGroupPolicy = &fileFSGroupPolicy
			},
		},
		{
			name: "StorageCapacity changed",
			modify: func(new *storage.CSIDriver) {
				new.Spec.StorageCapacity = &notStorageCapacity
			},
		},
		{
			name: "TokenRequests invalidated",
			modify: func(new *storage.CSIDriver) {
				new.Spec.TokenRequests = []storage.TokenRequest{{Audience: gcp}, {Audience: gcp}}
			},
		},
	}

	for _, test := range errorCases {
		t.Run(test.name, func(t *testing.T) {
			new := old.DeepCopy()
			test.modify(new)
			if errs := ValidateCSIDriverUpdate(new, &old); len(errs) == 0 {
				t.Errorf("Expected failure for test: %+v", new)
			}
		})
	}
}

func TestCSIDriverStorageCapacityEnablement(t *testing.T) {
	run := func(t *testing.T, enabled, withField bool) {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIStorageCapacity, enabled)()

		driverName := "test-driver"
		attachRequired := true
		podInfoOnMount := true
		requiresRepublish := true
		storageCapacity := true
		csiDriver := storage.CSIDriver{
			ObjectMeta: metav1.ObjectMeta{Name: driverName},
			Spec: storage.CSIDriverSpec{
				AttachRequired:    &attachRequired,
				PodInfoOnMount:    &podInfoOnMount,
				RequiresRepublish: &requiresRepublish,
			},
		}
		if withField {
			csiDriver.Spec.StorageCapacity = &storageCapacity
		}
		errs := ValidateCSIDriver(&csiDriver)
		success := !enabled || withField
		if success && len(errs) != 0 {
			t.Errorf("expected success, got: %v", errs)
		}
		if !success && len(errs) == 0 {
			t.Errorf("expected error, got success")
		}
	}

	yesNo := []bool{true, false}
	for _, enabled := range yesNo {
		t.Run(fmt.Sprintf("CSIStorageCapacity=%v", enabled), func(t *testing.T) {
			for _, withField := range yesNo {
				t.Run(fmt.Sprintf("with-field=%v", withField), func(t *testing.T) {
					run(t, enabled, withField)
				})
			}
		})
	}
}

func TestValidateCSIStorageCapacity(t *testing.T) {
	storageClassName := "test-sc"
	invalidName := "-invalid-@#$%^&*()-"

	goodCapacity := storage.CSIStorageCapacity{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "csc-329803da-fdd2-42e4-af6f-7b07e7ccc305",
			Namespace: metav1.NamespaceDefault,
		},
		StorageClassName: storageClassName,
	}
	goodTopology := metav1.LabelSelector{
		MatchLabels: map[string]string{"foo": "bar"},
	}

	scenarios := map[string]struct {
		isExpectedFailure bool
		capacity          *storage.CSIStorageCapacity
	}{
		"good-capacity": {
			capacity: &goodCapacity,
		},
		"missing-storage-class-name": {
			isExpectedFailure: true,
			capacity: func() *storage.CSIStorageCapacity {
				capacity := goodCapacity
				capacity.StorageClassName = ""
				return &capacity
			}(),
		},
		"bad-storage-class-name": {
			isExpectedFailure: true,
			capacity: func() *storage.CSIStorageCapacity {
				capacity := goodCapacity
				capacity.StorageClassName = invalidName
				return &capacity
			}(),
		},
		"good-capacity-value": {
			capacity: func() *storage.CSIStorageCapacity {
				capacity := goodCapacity
				capacity.Capacity = resource.NewQuantity(1, resource.BinarySI)
				return &capacity
			}(),
		},
		"bad-capacity-value": {
			isExpectedFailure: true,
			capacity: func() *storage.CSIStorageCapacity {
				capacity := goodCapacity
				capacity.Capacity = resource.NewQuantity(-11, resource.BinarySI)
				return &capacity
			}(),
		},
		"good-topology": {
			capacity: func() *storage.CSIStorageCapacity {
				capacity := goodCapacity
				capacity.NodeTopology = &goodTopology
				return &capacity
			}(),
		},
		"empty-topology": {
			capacity: func() *storage.CSIStorageCapacity {
				capacity := goodCapacity
				capacity.NodeTopology = &metav1.LabelSelector{}
				return &capacity
			}(),
		},
		"bad-topology-fields": {
			isExpectedFailure: true,
			capacity: func() *storage.CSIStorageCapacity {
				capacity := goodCapacity
				capacity.NodeTopology = &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "foo",
							Operator: metav1.LabelSelectorOperator("no-such-operator"),
							Values: []string{
								"bar",
							},
						},
					},
				}
				return &capacity
			}(),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateCSIStorageCapacity(scenario.capacity)
			if len(errs) == 0 && scenario.isExpectedFailure {
				t.Errorf("Unexpected success")
			}
			if len(errs) > 0 && !scenario.isExpectedFailure {
				t.Errorf("Unexpected failure: %+v", errs)
			}
		})
	}

}

func makeCSIServiceInValidCSIDriver() *storage.CSIDriver {
	driverName := "test-driver"
	gcp := "gcp"
	notRequiresRepublish := false
	return &storage.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{Name: driverName},
		Spec: storage.CSIDriverSpec{
			TokenRequests:     []storage.TokenRequest{{Audience: gcp}, {Audience: gcp}},
			RequiresRepublish: &notRequiresRepublish,
		},
	}
}

func makeCSIServiceCSIDriverCustom(tweaks ...func(csiDriver *storage.CSIDriver)) *storage.CSIDriver {
	csiDriver := makeCSIServiceInValidCSIDriver()
	for _, fn := range tweaks {
		fn(csiDriver)
	}
	return csiDriver
}

func TestCSIServiceAccountToken(t *testing.T) {
	aws := "aws"
	gcp := "gcp"
	tests := []struct {
		desc      string
		csiDriver *storage.CSIDriver
		wantErr   bool
	}{
		{
			desc:      "invalid - TokenRequests has tokens with the same audience",
			csiDriver: makeCSIServiceCSIDriverCustom(func(csiDriver *storage.CSIDriver) {
				csiDriver.Spec.TokenRequests = []storage.TokenRequest{{Audience: gcp, ExpirationSeconds: utilpointer.Int64Ptr(10)}}
			}),
			wantErr:   true,
		},
		{
			desc:      "invalid - TokenRequests has tokens with ExpirationSeconds less than 10min",
			csiDriver: makeCSIServiceCSIDriverCustom(),
			wantErr:   true,
		},
		{
			desc:      "invalid - TokenRequests has tokens with ExpirationSeconds less than 10min",
			csiDriver: makeCSIServiceCSIDriverCustom(func(csiDriver *storage.CSIDriver) {
				csiDriver.Spec.TokenRequests = []storage.TokenRequest{{Audience: gcp, ExpirationSeconds: utilpointer.Int64Ptr(1<<32 + 1)}}
			}),
			wantErr:   true,
		},
		{
			desc:      "valid - TokenRequests has at most one token with empty string audience",
			csiDriver: makeCSIServiceCSIDriverCustom(func(csiDriver *storage.CSIDriver) {
				csiDriver.Spec.TokenRequests = []storage.TokenRequest{{Audience: ""}}
			}),
		},
		{
			desc:      "valid - TokenRequests has tokens with different audience",
			csiDriver: makeCSIServiceCSIDriverCustom(func(csiDriver *storage.CSIDriver) {
				csiDriver.Spec.TokenRequests = []storage.TokenRequest{{}, {Audience: gcp}, {Audience: aws}}
			}),
		},
	}

	for _, test := range tests {
		test.csiDriver.Spec.AttachRequired = new(bool)
		test.csiDriver.Spec.PodInfoOnMount = new(bool)
		test.csiDriver.Spec.StorageCapacity = new(bool)
		if errs := ValidateCSIDriver(test.csiDriver); test.wantErr != (len(errs) != 0) {
			t.Errorf("ValidateCSIDriver = %v, want err: %v", errs, test.wantErr)
		}
	}
}
