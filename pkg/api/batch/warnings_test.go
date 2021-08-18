package batch

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/features"
)

func TestGetWarningsForJob(t *testing.T) {
	one := int32(1)
	hasBackingController := true
	testCases := []struct {
		name                   string
		job                    *batch.Job
		enableTTLAfterFinished bool
		expected               []string
	}{
		{
			name:     "null",
			job:      nil,
			expected: nil,
		},
		{
			name:                   "no ownerRef, TTL enabled, TTLfield not set, expect warning",
			enableTTLAfterFinished: true,
			job:                    &batch.Job{Spec: batch.JobSpec{}},
			expected:               []string{TTLWarningMsg},
		},
		{
			name:                   "no ownerRef, TTL enabled, TTLfield set, no warning expected",
			enableTTLAfterFinished: true,
			job:                    &batch.Job{Spec: batch.JobSpec{TTLSecondsAfterFinished: &one}},
			expected:               nil,
		},
		{
			name:                   "no ownerRef, TTL disabled, TTLfield not set, no warning expected",
			enableTTLAfterFinished: false,
			job:                    &batch.Job{Spec: batch.JobSpec{}},
			expected:               nil,
		},
		{
			name:                   "no ownerRef, TTL disabled, TTLfield set, no warning expected",
			enableTTLAfterFinished: false,
			job:                    &batch.Job{Spec: batch.JobSpec{TTLSecondsAfterFinished: &one}},
			expected:               nil,
		},
		{
			name:                   "ownerRef, TTL disabled, TTLfield set, no warning expected",
			enableTTLAfterFinished: false,
			job: &batch.Job{ObjectMeta: metav1.ObjectMeta{
				OwnerReferences: []metav1.OwnerReference{{Controller: &hasBackingController}}}, Spec: batch.JobSpec{
				TTLSecondsAfterFinished: &one}},
			expected: nil,
		},
		{
			name:                   "ownerRef, TTL disabled, TTLfield not set, no warning expected",
			enableTTLAfterFinished: false,
			job: &batch.Job{ObjectMeta: metav1.ObjectMeta{
				OwnerReferences: []metav1.OwnerReference{{Controller: &hasBackingController}}}, Spec: batch.JobSpec{}},
			expected: nil,
		},
		{
			name:                   "ownerRef, TTL enabled, TTLfield not set, no warning expected",
			enableTTLAfterFinished: true,
			job: &batch.Job{ObjectMeta: metav1.ObjectMeta{
				OwnerReferences: []metav1.OwnerReference{{Controller: &hasBackingController}}}, Spec: batch.JobSpec{}},
			expected: nil,
		},
		{
			name:                   "ownerRef, TTL enabled, TTLfield set, no warning expected",
			enableTTLAfterFinished: true,
			job: &batch.Job{ObjectMeta: metav1.ObjectMeta{
				OwnerReferences: []metav1.OwnerReference{{Controller: &hasBackingController}}}, Spec: batch.JobSpec{
				TTLSecondsAfterFinished: &one}},
			expected: nil,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TTLAfterFinished,
				tc.enableTTLAfterFinished)()
			actualWarnings := GetWarningsForJob(tc.job)
			if !reflect.DeepEqual(tc.expected, actualWarnings) {
				t.Errorf("expected %v, but got %v", tc.expected, actualWarnings)
			}
		})
	}
}
