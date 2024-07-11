/*
Copyright 2015 The Kubernetes Authors.

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

package job

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/batch"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

var ignoreErrValueDetail = cmpopts.IgnoreFields(field.Error{}, "BadValue", "Detail")

// TestJobStrategy_PrepareForUpdate tests various scenarios for PrepareForUpdate
func TestJobStrategy_PrepareForUpdate(t *testing.T) {
	validSelector := getValidLabelSelector()
	validPodTemplateSpec := getValidPodTemplateSpecForSelector(validSelector)

	podFailurePolicy := &batch.PodFailurePolicy{
		Rules: []batch.PodFailurePolicyRule{
			{
				Action: batch.PodFailurePolicyActionFailJob,
				OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
					ContainerName: ptr.To("container-name"),
					Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
					Values:        []int32{1},
				},
			},
		},
	}
	updatedPodFailurePolicy := &batch.PodFailurePolicy{
		Rules: []batch.PodFailurePolicyRule{
			{
				Action: batch.PodFailurePolicyActionIgnore,
				OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
					ContainerName: ptr.To("updated-container-name"),
					Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
					Values:        []int32{2},
				},
			},
		},
	}
	successPolicy := &batch.SuccessPolicy{
		Rules: []batch.SuccessPolicyRule{
			{
				SucceededIndexes: ptr.To("1,3-7"),
				SucceededCount:   ptr.To[int32](4),
			},
		},
	}
	updatedSuccessPolicy := &batch.SuccessPolicy{
		Rules: []batch.SuccessPolicyRule{
			{
				SucceededIndexes: ptr.To("1,3-7"),
				SucceededCount:   ptr.To[int32](5),
			},
		},
	}

	cases := map[string]struct {
		enableJobPodFailurePolicy     bool
		enableJobBackoffLimitPerIndex bool
		enableJobPodReplacementPolicy bool
		enableJobSuccessPolicy        bool
		job                           batch.Job
		updatedJob                    batch.Job
		wantJob                       batch.Job
	}{
		"update job with a new field; updated when JobSuccessPolicy enabled": {
			enableJobSuccessPolicy: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:      validSelector,
					Template:      validPodTemplateSpec,
					SuccessPolicy: nil,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:      validSelector,
					Template:      validPodTemplateSpec,
					SuccessPolicy: updatedSuccessPolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:      validSelector,
					Template:      validPodTemplateSpec,
					SuccessPolicy: updatedSuccessPolicy,
				},
			},
		},
		"update pre-existing field; updated when JobSuccessPolicy enabled": {
			enableJobSuccessPolicy: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:      validSelector,
					Template:      validPodTemplateSpec,
					SuccessPolicy: successPolicy,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:      validSelector,
					Template:      validPodTemplateSpec,
					SuccessPolicy: updatedSuccessPolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:      validSelector,
					Template:      validPodTemplateSpec,
					SuccessPolicy: updatedSuccessPolicy,
				},
			},
		},
		"update job with a new field: not update when JobSuccessPolicy disabled": {
			enableJobSuccessPolicy: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:      validSelector,
					Template:      validPodTemplateSpec,
					SuccessPolicy: nil,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:      validSelector,
					Template:      validPodTemplateSpec,
					SuccessPolicy: updatedSuccessPolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:      validSelector,
					Template:      validPodTemplateSpec,
					SuccessPolicy: nil,
				},
			},
		},
		"update pre-existing field; updated when JobSuccessPolicy disabled": {
			enableJobSuccessPolicy: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:      validSelector,
					Template:      validPodTemplateSpec,
					SuccessPolicy: successPolicy,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:      validSelector,
					Template:      validPodTemplateSpec,
					SuccessPolicy: updatedSuccessPolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:      validSelector,
					Template:      validPodTemplateSpec,
					SuccessPolicy: updatedSuccessPolicy,
				},
			},
		},
		"update job with a new field; updated when JobBackoffLimitPerIndex enabled": {
			enableJobBackoffLimitPerIndex: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					Template:             validPodTemplateSpec,
					BackoffLimitPerIndex: nil,
					MaxFailedIndexes:     nil,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					Template:             validPodTemplateSpec,
					BackoffLimitPerIndex: ptr.To[int32](1),
					MaxFailedIndexes:     ptr.To[int32](1),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					Template:             validPodTemplateSpec,
					BackoffLimitPerIndex: ptr.To[int32](1),
					MaxFailedIndexes:     ptr.To[int32](1),
				},
			},
		},
		"update job with a new field; not updated when JobBackoffLimitPerIndex disabled": {
			enableJobBackoffLimitPerIndex: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					Template:             validPodTemplateSpec,
					BackoffLimitPerIndex: nil,
					MaxFailedIndexes:     nil,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					Template:             validPodTemplateSpec,
					BackoffLimitPerIndex: ptr.To[int32](1),
					MaxFailedIndexes:     ptr.To[int32](1),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					Template:             validPodTemplateSpec,
					BackoffLimitPerIndex: nil,
					MaxFailedIndexes:     nil,
				},
			},
		},
		"update job with a new field; updated when JobPodFailurePolicy enabled": {
			enableJobPodFailurePolicy: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: nil,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
		},
		"update job with a new field; updated when JobPodReplacementPolicy enabled": {
			enableJobPodReplacementPolicy: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					Template:             validPodTemplateSpec,
					PodReplacementPolicy: nil,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					Template:             validPodTemplateSpec,
					PodReplacementPolicy: podReplacementPolicy(batch.Failed),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					Template:             validPodTemplateSpec,
					PodReplacementPolicy: podReplacementPolicy(batch.Failed),
				},
			},
		},
		"update job with a new field; not updated when JobPodReplacementPolicy disabled": {
			enableJobPodReplacementPolicy: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					Template:             validPodTemplateSpec,
					PodReplacementPolicy: nil,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					Template:             validPodTemplateSpec,
					PodReplacementPolicy: podReplacementPolicy(batch.Failed),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					Template:             validPodTemplateSpec,
					PodReplacementPolicy: nil,
				},
			},
		},
		"update job with a new field; not updated when JobPodFailurePolicy disabled": {
			enableJobPodFailurePolicy: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: nil,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: nil,
				},
			},
		},
		"update pre-existing field; updated when JobPodFailurePolicy enabled": {
			enableJobPodFailurePolicy: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: podFailurePolicy,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
		},
		"update pre-existing field; updated when JobPodFailurePolicy disabled": {
			enableJobPodFailurePolicy: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: podFailurePolicy,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: updatedPodFailurePolicy,
				},
			},
		},
		"add tracking annotation back": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					Template:         validPodTemplateSpec,
					PodFailurePolicy: podFailurePolicy,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
			},
		},
		"attempt status update and verify it doesn't change": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 2,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
		},
		"ensure generation doesn't change over non spec updates": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMetaWithAnnotations(0, map[string]string{"hello": "world"}),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 2,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMetaWithAnnotations(0, map[string]string{"hello": "world"}),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
		},
		"test updating suspend false->true": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
					Suspend:  ptr.To(false),
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMetaWithAnnotations(0, map[string]string{"hello": "world"}),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
					Suspend:  ptr.To(true),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMetaWithAnnotations(1, map[string]string{"hello": "world"}),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
					Suspend:  ptr.To(true),
				},
			},
		},
		"test updating suspend nil -> true": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
				},
			},
			updatedJob: batch.Job{
				ObjectMeta: getValidObjectMetaWithAnnotations(0, map[string]string{"hello": "world"}),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
					Suspend:  ptr.To(true),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMetaWithAnnotations(1, map[string]string{"hello": "world"}),
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: validPodTemplateSpec,
					Suspend:  ptr.To(true),
				},
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobPodFailurePolicy, tc.enableJobPodFailurePolicy)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobBackoffLimitPerIndex, tc.enableJobBackoffLimitPerIndex)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobPodReplacementPolicy, tc.enableJobPodReplacementPolicy)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobSuccessPolicy, tc.enableJobSuccessPolicy)
			ctx := genericapirequest.NewDefaultContext()

			Strategy.PrepareForUpdate(ctx, &tc.updatedJob, &tc.job)

			if diff := cmp.Diff(tc.wantJob, tc.updatedJob); diff != "" {
				t.Errorf("Job update differences (-want,+got):\n%s", diff)
			}
		})
	}
}

// TestJobStrategy_PrepareForCreate tests various scenarios for PrepareForCreate
func TestJobStrategy_PrepareForCreate(t *testing.T) {
	validSelector := getValidLabelSelector()
	validPodTemplateSpec := getValidPodTemplateSpecForSelector(validSelector)
	validSelectorWithBatchLabels := &metav1.LabelSelector{MatchLabels: getValidBatchLabelsWithNonBatch()}
	expectedPodTemplateSpec := getValidPodTemplateSpecForSelector(validSelectorWithBatchLabels)

	podFailurePolicy := &batch.PodFailurePolicy{
		Rules: []batch.PodFailurePolicyRule{
			{
				Action: batch.PodFailurePolicyActionFailJob,
				OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
					ContainerName: ptr.To("container-name"),
					Operator:      batch.PodFailurePolicyOnExitCodesOpIn,
					Values:        []int32{1},
				},
			},
		},
	}
	successPolicy := &batch.SuccessPolicy{
		Rules: []batch.SuccessPolicyRule{
			{
				SucceededIndexes: ptr.To("1,3-7"),
				SucceededCount:   ptr.To[int32](4),
			},
		},
	}

	cases := map[string]struct {
		enableJobPodFailurePolicy     bool
		enableJobBackoffLimitPerIndex bool
		enableJobPodReplacementPolicy bool
		enableJobManageBy             bool
		enableJobSuccessPolicy        bool
		job                           batch.Job
		wantJob                       batch.Job
	}{
		"generate selectors": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       validPodTemplateSpec,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       expectedPodTemplateSpec,
				},
			},
		},
		"create job with a new field; JobSuccessPolicy enabled": {
			enableJobSuccessPolicy: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       validPodTemplateSpec,
					SuccessPolicy:  successPolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       expectedPodTemplateSpec,
					SuccessPolicy:  successPolicy,
				},
			},
		},
		"create job with a new field; JobSuccessPolicy disabled": {
			enableJobSuccessPolicy: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       validPodTemplateSpec,
					SuccessPolicy:  successPolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       validPodTemplateSpec,
					SuccessPolicy:  nil,
				},
			},
		},
		"create job with a new fields; JobBackoffLimitPerIndex enabled": {
			enableJobBackoffLimitPerIndex: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					ManualSelector:       ptr.To(false),
					Template:             validPodTemplateSpec,
					BackoffLimitPerIndex: ptr.To[int32](1),
					MaxFailedIndexes:     ptr.To[int32](1),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					ManualSelector:       ptr.To(false),
					Template:             expectedPodTemplateSpec,
					BackoffLimitPerIndex: ptr.To[int32](1),
					MaxFailedIndexes:     ptr.To[int32](1),
				},
			},
		},
		"create job with a new fields; JobBackoffLimitPerIndex disabled": {
			enableJobBackoffLimitPerIndex: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					ManualSelector:       ptr.To(false),
					Template:             validPodTemplateSpec,
					BackoffLimitPerIndex: ptr.To[int32](1),
					MaxFailedIndexes:     ptr.To[int32](1),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					ManualSelector:       ptr.To(false),
					Template:             expectedPodTemplateSpec,
					BackoffLimitPerIndex: nil,
					MaxFailedIndexes:     nil,
				},
			},
		},
		"create job with a new field; JobPodFailurePolicy enabled": {
			enableJobPodFailurePolicy: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					ManualSelector:   ptr.To(false),
					Template:         validPodTemplateSpec,
					PodFailurePolicy: podFailurePolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					ManualSelector:   ptr.To(false),
					Template:         expectedPodTemplateSpec,
					PodFailurePolicy: podFailurePolicy,
				},
			},
		},
		"create job with a new field; JobPodReplacementPolicy enabled": {
			enableJobPodReplacementPolicy: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					ManualSelector:       ptr.To(false),
					Template:             validPodTemplateSpec,
					PodReplacementPolicy: podReplacementPolicy(batch.Failed),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					ManualSelector:       ptr.To(false),
					Template:             expectedPodTemplateSpec,
					PodReplacementPolicy: podReplacementPolicy(batch.Failed),
				},
			},
		},
		"create job with a new field; JobPodReplacementPolicy disabled": {
			enableJobPodReplacementPolicy: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					ManualSelector:       ptr.To(false),
					Template:             validPodTemplateSpec,
					PodReplacementPolicy: podReplacementPolicy(batch.Failed),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					ManualSelector:       ptr.To(false),
					Template:             expectedPodTemplateSpec,
					PodReplacementPolicy: nil,
				},
			},
		},
		"create job with a new field; JobPodFailurePolicy disabled": {
			enableJobPodFailurePolicy: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					ManualSelector:   ptr.To(false),
					Template:         validPodTemplateSpec,
					PodFailurePolicy: podFailurePolicy,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:         validSelector,
					ManualSelector:   ptr.To(false),
					Template:         expectedPodTemplateSpec,
					PodFailurePolicy: nil,
				},
			},
		},
		"job does not allow setting status on create": {
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       validPodTemplateSpec,
				},
				Status: batch.JobStatus{
					Active: 1,
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       expectedPodTemplateSpec,
				},
			},
		},
		"create job with pod failure policy using FailIndex action; JobPodFailurePolicy enabled, JobBackoffLimitPerIndex disabled": {
			enableJobBackoffLimitPerIndex: false,
			enableJobPodFailurePolicy:     true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					ManualSelector:       ptr.To(false),
					Template:             validPodTemplateSpec,
					BackoffLimitPerIndex: ptr.To[int32](1),
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionFailIndex,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
						},
					},
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       expectedPodTemplateSpec,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{},
					},
				},
			},
		},
		"create job with multiple pod failure policy rules, some using FailIndex action; JobPodFailurePolicy enabled, JobBackoffLimitPerIndex disabled": {
			enableJobBackoffLimitPerIndex: false,
			enableJobPodFailurePolicy:     true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:             validSelector,
					ManualSelector:       ptr.To(false),
					Template:             validPodTemplateSpec,
					BackoffLimitPerIndex: ptr.To[int32](1),
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionFailJob,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{2},
								},
							},
							{
								Action: batch.PodFailurePolicyActionFailIndex,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
							{
								Action: batch.PodFailurePolicyActionIgnore,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{13},
								},
							},
						},
					},
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       expectedPodTemplateSpec,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionFailJob,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{2},
								},
							},
							{
								Action: batch.PodFailurePolicyActionIgnore,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{13},
								},
							},
						},
					},
				},
			},
		},
		"managedBy field is dropped when the feature gate is disabled": {
			enableJobManageBy: false,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       validPodTemplateSpec,
					ManagedBy:      ptr.To("custom-controller-name"),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       expectedPodTemplateSpec,
				},
			},
		},
		"managedBy field is set when the feature gate is enabled": {
			enableJobManageBy: true,
			job: batch.Job{
				ObjectMeta: getValidObjectMeta(0),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       validPodTemplateSpec,
					ManagedBy:      ptr.To("custom-controller-name"),
				},
			},
			wantJob: batch.Job{
				ObjectMeta: getValidObjectMeta(1),
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(false),
					Template:       expectedPodTemplateSpec,
					ManagedBy:      ptr.To("custom-controller-name"),
				},
			},
		},
	}

	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobPodFailurePolicy, tc.enableJobPodFailurePolicy)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobBackoffLimitPerIndex, tc.enableJobBackoffLimitPerIndex)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobPodReplacementPolicy, tc.enableJobPodReplacementPolicy)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobManagedBy, tc.enableJobManageBy)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobSuccessPolicy, tc.enableJobSuccessPolicy)
			ctx := genericapirequest.NewDefaultContext()

			Strategy.PrepareForCreate(ctx, &tc.job)

			if diff := cmp.Diff(tc.wantJob, tc.job); diff != "" {
				t.Errorf("Job pod failure policy (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestJobStrategy_GarbageCollectionPolicy(t *testing.T) {
	// Make sure we correctly implement the interface.
	// Otherwise a typo could silently change the default.
	var gcds rest.GarbageCollectionDeleteStrategy = Strategy
	if got, want := gcds.DefaultGarbageCollectionPolicy(genericapirequest.NewContext()), rest.DeleteDependents; got != want {
		t.Errorf("DefaultGarbageCollectionPolicy() = %#v, want %#v", got, want)
	}

	var (
		v1Ctx           = genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v1", Resource: "jobs"})
		otherVersionCtx = genericapirequest.WithRequestInfo(genericapirequest.NewContext(), &genericapirequest.RequestInfo{APIGroup: "batch", APIVersion: "v100", Resource: "jobs"})
	)
	if got, want := gcds.DefaultGarbageCollectionPolicy(v1Ctx), rest.OrphanDependents; got != want {
		t.Errorf("DefaultGarbageCollectionPolicy() = %#v, want %#v", got, want)
	}
	if got, want := gcds.DefaultGarbageCollectionPolicy(otherVersionCtx), rest.DeleteDependents; got != want {
		t.Errorf("DefaultGarbageCollectionPolicy() = %#v, want %#v", got, want)
	}
}

func TestJobStrategy_ValidateUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyOnFailure)),
	}
	validPodTemplateSpecNever := *validPodTemplateSpec.DeepCopy()
	validPodTemplateSpecNever.Spec.RestartPolicy = api.RestartPolicyNever
	now := metav1.Now()
	cases := map[string]struct {
		enableJobPodFailurePolicy     bool
		enableJobBackoffLimitPerIndex bool
		job                           *batch.Job
		update                        func(*batch.Job)
		wantErrs                      field.ErrorList
	}{
		"update parallelism": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Parallelism = ptr.To[int32](2)
			},
		},
		"update completions disallowed": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
					Completions:    ptr.To[int32](1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Completions = ptr.To[int32](2)
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "spec.completions"},
			},
		},
		"preserving tracking annotation": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations: map[string]string{
						batch.JobTrackingFinalizer: "",
					},
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},
			update: func(job *batch.Job) {
				job.Annotations["foo"] = "bar"
			},
		},
		"deleting user annotation": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations: map[string]string{
						batch.JobTrackingFinalizer: "",
						"foo":                      "bar",
					},
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},
			update: func(job *batch.Job) {
				delete(job.Annotations, "foo")
			},
		},
		"updating node selector for unsuspended job disallowed": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.NodeSelector = map[string]string{"foo": "bar"}
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "spec.template"},
			},
		},
		"updating node selector for suspended but previously started job disallowed": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
					Suspend:        ptr.To(true),
				},
				Status: batch.JobStatus{
					StartTime: &now,
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.NodeSelector = map[string]string{"foo": "bar"}
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "spec.template"},
			},
		},
		"updating node selector for suspended and not previously started job allowed": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
					Suspend:        ptr.To(true),
				},
			},
			update: func(job *batch.Job) {
				job.Spec.Template.Spec.NodeSelector = map[string]string{"foo": "bar"}
			},
		},
		"invalid label selector": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels:      map[string]string{"a": "b"},
						MatchExpressions: []metav1.LabelSelectorRequirement{{Key: "key", Operator: metav1.LabelSelectorOpNotIn, Values: []string{"bad value"}}},
					},
					ManualSelector: ptr.To(true),
					Template:       validPodTemplateSpec,
				},
			},
			update: func(job *batch.Job) {
				job.Annotations["hello"] = "world"
			},
		},
		"old job has no batch.kubernetes.io labels": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					UID:             "test",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "10",
					Annotations:     map[string]string{"hello": "world"},
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{batch.LegacyControllerUidLabel: "test"},
					},
					Parallelism: ptr.To[int32](4),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{batch.LegacyJobNameLabel: "myjob", batch.LegacyControllerUidLabel: "test"},
						},
						Spec: podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyOnFailure)),
					},
				},
			},
			update: func(job *batch.Job) {
				job.Annotations["hello"] = "world"
			},
		},
		"old job has all labels": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					UID:             "test",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "10",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{batch.ControllerUidLabel: "test"},
					},
					Parallelism: ptr.To[int32](4),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{batch.LegacyJobNameLabel: "myjob", batch.JobNameLabel: "myjob", batch.LegacyControllerUidLabel: "test", batch.ControllerUidLabel: "test"},
						},
						Spec: podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyOnFailure)),
					},
				},
			},
			update: func(job *batch.Job) {
				job.Annotations["hello"] = "world"
			},
		},
		"old job is using FailIndex JobBackoffLimitPerIndex is disabled, but FailIndex was already used": {
			enableJobPodFailurePolicy:     true,
			enableJobBackoffLimitPerIndex: false,
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Annotations:     map[string]string{"foo": "bar"},
				},
				Spec: batch.JobSpec{
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](1),
					Selector:             validSelector,
					ManualSelector:       ptr.To(true),
					Template:             validPodTemplateSpecNever,
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionFailIndex,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
						},
					},
				},
			},
			update: func(job *batch.Job) {
				job.Annotations["hello"] = "world"
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobPodFailurePolicy, tc.enableJobPodFailurePolicy)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobBackoffLimitPerIndex, tc.enableJobBackoffLimitPerIndex)
			newJob := tc.job.DeepCopy()
			tc.update(newJob)
			errs := Strategy.ValidateUpdate(ctx, newJob, tc.job)
			if diff := cmp.Diff(tc.wantErrs, errs, ignoreErrValueDetail); diff != "" {
				t.Errorf("Unexpected errors (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestJobStrategy_WarningsOnUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyOnFailure)),
	}
	cases := map[string]struct {
		oldJob            *batch.Job
		job               *batch.Job
		wantWarningsCount int32
	}{
		"generation 0 for both": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      0,
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},

			oldJob: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      0,
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},
		},
		"generation 1 for new; force WarningsOnUpdate to check PodTemplate for updates": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      1,
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},

			oldJob: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      0,
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},
		},
		"force validation failure in pod template": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      1,
				},
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: api.PodTemplateSpec{
						Spec: api.PodSpec{ImagePullSecrets: []api.LocalObjectReference{{Name: ""}}},
					},
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},

			oldJob: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Generation:      0,
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},
			wantWarningsCount: 1,
		},
		"Invalid transition to high parallelism": {
			wantWarningsCount: 1,
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob2",
					Namespace:       metav1.NamespaceDefault,
					Generation:      1,
					ResourceVersion: "0",
				},
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](100_001),
					Parallelism:    ptr.To[int32](10_001),
					Template:       validPodTemplateSpec,
				},
			},
			oldJob: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob2",
					Namespace:       metav1.NamespaceDefault,
					Generation:      0,
					ResourceVersion: "0",
				},
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](100_001),
					Parallelism:    ptr.To[int32](10_000),
					Template:       validPodTemplateSpec,
				},
			},
		},
	}
	for val, tc := range cases {
		t.Run(val, func(t *testing.T) {
			gotWarnings := Strategy.WarningsOnUpdate(ctx, tc.job, tc.oldJob)
			if len(gotWarnings) != int(tc.wantWarningsCount) {
				t.Errorf("got warning length of %d but expected %d", len(gotWarnings), tc.wantWarningsCount)
			}
		})
	}
}
func TestJobStrategy_WarningsOnCreate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	theUID := types.UID("1a2b3c4d5e6f7g8h9i0k")
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplate := api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyOnFailure)),
	}
	validSpec := batch.JobSpec{
		CompletionMode: completionModePtr(batch.NonIndexedCompletion),
		Selector:       nil,
		Template:       validPodTemplate,
	}

	testcases := map[string]struct {
		job               *batch.Job
		wantWarningsCount int32
	}{
		"happy path job": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob2",
					Namespace: metav1.NamespaceDefault,
					UID:       theUID,
				},
				Spec: validSpec,
			},
		},
		"dns invalid name": {
			wantWarningsCount: 1,
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "my job2",
					Namespace: metav1.NamespaceDefault,
					UID:       theUID,
				},
				Spec: validSpec,
			},
		},
		"high completions and parallelism": {
			wantWarningsCount: 1,
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob2",
					Namespace: metav1.NamespaceDefault,
					UID:       theUID,
				},
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Parallelism:    ptr.To[int32](100_001),
					Completions:    ptr.To[int32](100_001),
					Template:       validPodTemplate,
				},
			},
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			gotWarnings := Strategy.WarningsOnCreate(ctx, tc.job)
			if len(gotWarnings) != int(tc.wantWarningsCount) {
				t.Errorf("got warning length of %d but expected %d", len(gotWarnings), tc.wantWarningsCount)
			}
		})
	}
}
func TestJobStrategy_Validate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()

	theUID := getValidUID()
	validSelector := getValidLabelSelector()
	batchLabels := getValidBatchLabels()
	labelsWithNonBatch := getValidBatchLabelsWithNonBatch()
	defaultSelector := &metav1.LabelSelector{MatchLabels: map[string]string{batch.ControllerUidLabel: string(theUID)}}
	validPodSpec := podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyOnFailure))
	validPodSpecNever := podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyNever))
	validObjectMeta := getValidObjectMeta(0)
	testcases := map[string]struct {
		enableJobPodFailurePolicy     bool
		enableJobBackoffLimitPerIndex bool
		job                           *batch.Job
		wantJob                       *batch.Job
		wantWarningCount              int32
	}{
		"valid job with batch labels in pod template": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       defaultSelector,
					ManualSelector: ptr.To(false),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: batchLabels,
						},
						Spec: validPodSpec,
					}},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       defaultSelector,
					ManualSelector: ptr.To(false),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: batchLabels,
						},
						Spec: validPodSpec,
					}},
			},
		},
		"valid job with batch and non-batch labels in pod template": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       defaultSelector,
					ManualSelector: ptr.To(false),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: labelsWithNonBatch,
						},
						Spec: validPodSpec,
					}},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       defaultSelector,
					ManualSelector: ptr.To(false),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: labelsWithNonBatch,
						},
						Spec: validPodSpec,
					}},
			},
		},
		"job with non-batch labels and without batch labels in pod template": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       defaultSelector,
					ManualSelector: ptr.To(false),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{},
						},
						Spec: validPodSpec,
					}},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       defaultSelector,
					ManualSelector: ptr.To(false),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{},
						},
						Spec: validPodSpec,
					}},
			},
			wantWarningCount: 5,
		},
		"no labels in job": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: defaultSelector,
					Template: api.PodTemplateSpec{
						Spec: validPodSpec,
					}},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: defaultSelector,
					Template: api.PodTemplateSpec{
						Spec: validPodSpec,
					}},
			},
			wantWarningCount: 5,
		},
		"manual selector; do not generate labels": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpec,
					},
					Completions:    ptr.To[int32](2),
					ManualSelector: ptr.To(true),
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector: validSelector,
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpec,
					},
					Completions:    ptr.To[int32](2),
					ManualSelector: ptr.To(true),
				},
			},
		},
		"valid job with extended configuration": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       defaultSelector,
					ManualSelector: ptr.To(false),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: labelsWithNonBatch,
						},
						Spec: validPodSpec,
					},
					Completions:             ptr.To[int32](2),
					Suspend:                 ptr.To(true),
					TTLSecondsAfterFinished: ptr.To[int32](0),
					CompletionMode:          completionModePtr(batch.IndexedCompletion),
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       defaultSelector,
					ManualSelector: ptr.To(false),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: labelsWithNonBatch,
						},
						Spec: validPodSpec,
					},
					Completions:             ptr.To[int32](2),
					Suspend:                 ptr.To(true),
					TTLSecondsAfterFinished: ptr.To[int32](0),
					CompletionMode:          completionModePtr(batch.IndexedCompletion),
				},
			},
		},
		"fail validation due to invalid volume spec": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       defaultSelector,
					ManualSelector: ptr.To(false),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: labelsWithNonBatch,
						},
						Spec: podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyOnFailure),
							podtest.SetVolumes(api.Volume{Name: "volume-name"})),
					},
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       defaultSelector,
					ManualSelector: ptr.To(false),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: labelsWithNonBatch,
						},
						Spec: podtest.MakePodSpec(podtest.SetRestartPolicy(api.RestartPolicyOnFailure),
							podtest.SetVolumes(api.Volume{Name: "volume-name"})),
					},
				},
			},
			wantWarningCount: 1,
		},
		"FailIndex action; when JobBackoffLimitPerIndex is disabled - validation error": {
			enableJobPodFailurePolicy:     true,
			enableJobBackoffLimitPerIndex: false,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpecNever,
					},
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionFailIndex,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
						},
					},
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpecNever,
					},
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionFailIndex,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
						},
					},
				},
			},
			wantWarningCount: 1,
		},
		"FailIndex action; when JobBackoffLimitPerIndex is enabled, but not used - validation error": {
			enableJobPodFailurePolicy:     true,
			enableJobBackoffLimitPerIndex: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpecNever,
					},
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionFailIndex,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
						},
					},
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:       validSelector,
					ManualSelector: ptr.To(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpecNever,
					},
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionFailIndex,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
						},
					},
				},
			},
			wantWarningCount: 1,
		},
		"FailIndex action; when JobBackoffLimitPerIndex is enabled and used - no error": {
			enableJobPodFailurePolicy:     true,
			enableJobBackoffLimitPerIndex: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](1),
					Selector:             validSelector,
					ManualSelector:       ptr.To(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpecNever,
					},
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionFailIndex,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
						},
					},
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					Completions:          ptr.To[int32](2),
					BackoffLimitPerIndex: ptr.To[int32](1),
					Selector:             validSelector,
					ManualSelector:       ptr.To(true),
					Template: api.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: validSelector.MatchLabels,
						},
						Spec: validPodSpecNever,
					},
					PodFailurePolicy: &batch.PodFailurePolicy{
						Rules: []batch.PodFailurePolicyRule{
							{
								Action: batch.PodFailurePolicyActionFailIndex,
								OnExitCodes: &batch.PodFailurePolicyOnExitCodesRequirement{
									Operator: batch.PodFailurePolicyOnExitCodesOpIn,
									Values:   []int32{1},
								},
							},
						},
					},
				},
			},
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobPodFailurePolicy, tc.enableJobPodFailurePolicy)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobBackoffLimitPerIndex, tc.enableJobBackoffLimitPerIndex)
			errs := Strategy.Validate(ctx, tc.job)
			if len(errs) != int(tc.wantWarningCount) {
				t.Errorf("want warnings %d but got %d, errors: %v", tc.wantWarningCount, len(errs), errs)
			}
			if diff := cmp.Diff(tc.wantJob, tc.job); diff != "" {
				t.Errorf("Unexpected job (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestStrategy_ResetFields(t *testing.T) {
	resetFields := Strategy.GetResetFields()
	if len(resetFields) != 1 {
		t.Errorf("ResetFields should have 1 element, but have %d", len(resetFields))
	}
}

func TestJobStatusStrategy_ResetFields(t *testing.T) {
	resetFields := StatusStrategy.GetResetFields()
	if len(resetFields) != 1 {
		t.Errorf("ResetFields should have 1 element, but have %d", len(resetFields))
	}
}

func TestStatusStrategy_PrepareForUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
	validObjectMeta := metav1.ObjectMeta{
		Name:            "myjob",
		Namespace:       metav1.NamespaceDefault,
		ResourceVersion: "10",
	}

	cases := map[string]struct {
		job     *batch.Job
		newJob  *batch.Job
		wantJob *batch.Job
	}{
		"job must allow status updates": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: ptr.To[int32](4),
				},
				Status: batch.JobStatus{
					Active: 11,
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: ptr.To[int32](4),
				},
				Status: batch.JobStatus{
					Active: 12,
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: ptr.To[int32](4),
				},
				Status: batch.JobStatus{
					Active: 12,
				},
			},
		},
		"parallelism changes not allowed": {
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: ptr.To[int32](3),
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: ptr.To[int32](4),
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: ptr.To[int32](3),
				},
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			StatusStrategy.PrepareForUpdate(ctx, tc.newJob, tc.job)
			if diff := cmp.Diff(tc.wantJob, tc.newJob); diff != "" {
				t.Errorf("Unexpected job (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestStatusStrategy_ValidateUpdate(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
	validObjectMeta := metav1.ObjectMeta{
		Name:            "myjob",
		Namespace:       metav1.NamespaceDefault,
		ResourceVersion: "10",
	}
	validSuccessPolicy := &batch.SuccessPolicy{
		Rules: []batch.SuccessPolicyRule{{
			SucceededIndexes: ptr.To("0-2"),
		}},
	}
	now := metav1.Now()
	nowPlusMinute := metav1.Time{Time: now.Add(time.Minute)}

	cases := map[string]struct {
		enableJobManagedBy            bool
		enableJobSuccessPolicy        bool
		enableJobPodReplacementPolicy bool

		job      *batch.Job
		newJob   *batch.Job
		wantJob  *batch.Job
		wantErrs field.ErrorList
	}{
		"incoming resource version on update should not be mutated": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "10",
				},
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: ptr.To[int32](4),
				},
			},
			newJob: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "9",
				},
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: ptr.To[int32](4),
				},
			},
			wantJob: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "9",
				},
				Spec: batch.JobSpec{
					Selector:    validSelector,
					Template:    validPodTemplateSpec,
					Parallelism: ptr.To[int32](4),
				},
			},
		},
		"invalid addition of both Failed=True and Complete=True; allowed because feature gate disabled": {
			enableJobManagedBy: false,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &now,
					CompletionTime: &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
						{
							Type:   batch.JobFailed,
							Status: api.ConditionTrue,
						},
					},
				},
			},
		},
		"invalid addition of both Failed=True and Complete=True": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &now,
					CompletionTime: &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
						{
							Type:   batch.JobFailed,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"completionTime can be removed to fix still running job": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &now,
					CompletionTime: &now,
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime: &now,
				},
			},
		},
		"invalid attempt to transition to Failed=True without startTime": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobFailed,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "status.startTime"},
			},
		},
		"invalid attempt to transition to Complete=True without startTime": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					CompletionTime: &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "status.startTime"},
			},
		},
		"invalid attempt to transition to Complete=True with active > 0": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &now,
					CompletionTime: &now,
					Active:         1,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.active"},
			},
		},
		"transition to Failed condition with terminating>0 and ready>0": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime: &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobFailed,
							Status: api.ConditionTrue,
						},
					},
					Terminating: ptr.To[int32](1),
					Ready:       ptr.To[int32](1),
				},
			},
		},
		"invalid attempt to transition to Failed=True with uncountedTerminatedPods.Failed>0": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime: &now,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Failed: []types.UID{"a"},
					},
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobFailed,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.uncountedTerminatedPods"},
			},
		},
		"invalid attempt to update uncountedTerminatedPods.Succeeded for Complete job": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &now,
					CompletionTime: &now,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Failed: []types.UID{"a"},
					},
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &now,
					CompletionTime: &now,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Failed: []types.UID{"b"},
					},
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.uncountedTerminatedPods"},
			},
		},
		"non-empty uncountedTerminatedPods for complete job, unrelated update": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &now,
					CompletionTime: &now,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Failed: []types.UID{"a"},
					},
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &now,
					CompletionTime: &now,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Failed: []types.UID{"a"},
					},
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
						{
							Type:   batch.JobConditionType("CustomJobCondition"),
							Status: api.ConditionTrue,
						},
					},
				},
			},
		},
		"invalid attempt to transition to Complete=True with uncountedTerminatedPods.Succeeded>0": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &now,
					CompletionTime: &now,
					UncountedTerminatedPods: &batch.UncountedTerminatedPods{
						Succeeded: []types.UID{"a"},
					},
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.uncountedTerminatedPods"},
			},
		},
		"invalid addition Complete=True without setting CompletionTime": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime: &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "status.completionTime"},
			},
		},
		"invalid attempt to remove completionTime": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					CompletionTime: &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					CompletionTime: nil,
					StartTime:      &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "status.completionTime"},
			},
		},
		"verify startTime can be cleared for suspended job": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Suspend: ptr.To(true),
				},
				Status: batch.JobStatus{
					StartTime: &now,
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Suspend: ptr.To(true),
				},
				Status: batch.JobStatus{
					StartTime: nil,
				},
			},
		},
		"verify startTime cannot be removed for unsuspended job": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime: &now,
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime: nil,
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "status.startTime"},
			},
		},
		"verify startTime cannot be updated for unsuspended job": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime: &now,
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime: &nowPlusMinute,
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeRequired, Field: "status.startTime"},
			},
		},
		"invalid attempt to set completionTime before startTime": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime: &nowPlusMinute,
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &nowPlusMinute,
					CompletionTime: &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.completionTime"},
			},
		},
		"invalid attempt to modify completionTime": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					CompletionTime: &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					CompletionTime: &nowPlusMinute,
					StartTime:      &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.completionTime"},
			},
		},
		"invalid removal of terminal condition Failed=True": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobFailed,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"invalid removal of terminal condition Complete=True": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"invalid removal of terminal condition FailureTarget=True": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobFailureTarget,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"invalid addition of FailureTarget=True when Complete=True": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &now,
					CompletionTime: &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &now,
					CompletionTime: &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
						{
							Type:   batch.JobFailureTarget,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"invalid attempt setting of CompletionTime when there is no Complete condition": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					CompletionTime: &now,
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.completionTime"},
			},
		},
		"invalid CompletionTime when there is no Complete condition, but allowed": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					CompletionTime: &now,
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					CompletionTime: &now,
					Active:         1,
				},
			},
		},
		"invalid attempt setting CompletedIndexes when non-indexed completion mode is used": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.NonIndexedCompletion),
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.NonIndexedCompletion),
				},
				Status: batch.JobStatus{
					StartTime:        &now,
					CompletedIndexes: "0",
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.completedIndexes"},
			},
		},
		"invalid because CompletedIndexes set when non-indexed completion mode is used; but allowed": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.NonIndexedCompletion),
				},
				Status: batch.JobStatus{
					CompletedIndexes: "0",
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.NonIndexedCompletion),
				},
				Status: batch.JobStatus{
					CompletedIndexes: "0",
					Active:           1,
				},
			},
		},
		"invalid attempt setting FailedIndexes when not backoffLimitPerIndex": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
				Status: batch.JobStatus{
					FailedIndexes: ptr.To("0"),
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.failedIndexes"},
			},
		},
		"invalid attempt to decrease the failed counter": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions: ptr.To[int32](5),
				},
				Status: batch.JobStatus{
					Failed: 3,
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions: ptr.To[int32](5),
				},
				Status: batch.JobStatus{
					Failed: 1,
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.failed"},
			},
		},
		"invalid attempt to decrease the succeeded counter": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions: ptr.To[int32](5),
				},
				Status: batch.JobStatus{
					Succeeded: 3,
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions: ptr.To[int32](5),
				},
				Status: batch.JobStatus{
					Succeeded: 1,
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.succeeded"},
			},
		},
		"invalid attempt to set bad format for CompletedIndexes": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
				Status: batch.JobStatus{
					CompletedIndexes: "invalid format",
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.completedIndexes"},
			},
		},
		"invalid format for CompletedIndexes, but allowed": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
				Status: batch.JobStatus{
					CompletedIndexes: "invalid format",
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
				Status: batch.JobStatus{
					CompletedIndexes: "invalid format",
					Active:           1,
				},
			},
		},
		"invalid attempt to set bad format for FailedIndexes": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](5),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					BackoffLimitPerIndex: ptr.To[int32](1),
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](5),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					BackoffLimitPerIndex: ptr.To[int32](1),
				},
				Status: batch.JobStatus{
					FailedIndexes: ptr.To("invalid format"),
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.failedIndexes"},
			},
		},
		"invalid format for FailedIndexes, but allowed": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](5),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					BackoffLimitPerIndex: ptr.To[int32](1),
				},
				Status: batch.JobStatus{
					FailedIndexes: ptr.To("invalid format"),
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:          ptr.To[int32](5),
					CompletionMode:       completionModePtr(batch.IndexedCompletion),
					BackoffLimitPerIndex: ptr.To[int32](1),
				},
				Status: batch.JobStatus{
					FailedIndexes: ptr.To("invalid format"),
					Active:        1,
				},
			},
		},
		"more ready pods than active, but allowed": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions: ptr.To[int32](5),
				},
				Status: batch.JobStatus{
					Active: 1,
					Ready:  ptr.To[int32](2),
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions: ptr.To[int32](5),
				},
				Status: batch.JobStatus{
					Active:    1,
					Ready:     ptr.To[int32](2),
					Succeeded: 1,
				},
			},
		},
		"invalid addition of both FailureTarget=True and Complete=True": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					StartTime:      &now,
					CompletionTime: &now,
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
						{
							Type:   batch.JobFailureTarget,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"invalid failedIndexes, which overlap with completedIndexes": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
				Status: batch.JobStatus{
					FailedIndexes:    ptr.To("0,2"),
					CompletedIndexes: "3-4",
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
				Status: batch.JobStatus{
					FailedIndexes:    ptr.To("0,2"),
					CompletedIndexes: "2-4",
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.failedIndexes"},
			},
		},
		"failedIndexes overlap with completedIndexes, unrelated field change": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
				Status: batch.JobStatus{
					FailedIndexes:    ptr.To("0,2"),
					CompletedIndexes: "2-4",
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					Completions:    ptr.To[int32](5),
					CompletionMode: completionModePtr(batch.IndexedCompletion),
				},
				Status: batch.JobStatus{
					FailedIndexes:    ptr.To("0,2"),
					CompletedIndexes: "2-4",
					Active:           1,
				},
			},
		},
		"invalid addition of SuccessCriteriaMet for NonIndexed Job": {
			enableJobSuccessPolicy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					SuccessPolicy: validSuccessPolicy,
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					SuccessPolicy: validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobSuccessCriteriaMet,
						Status: api.ConditionTrue,
					}},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"valid update of Job if SuccessCriteriaMet already present for NonIndexed Jobs; JobSuccessPolicy enabled, while JobManagedBy and JobPodReplacementPolicy disabled": {
			enableJobSuccessPolicy:        true,
			enableJobManagedBy:            false,
			enableJobPodReplacementPolicy: false,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec:       batch.JobSpec{},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobSuccessCriteriaMet,
						Status: api.ConditionTrue,
					}},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec:       batch.JobSpec{},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobSuccessCriteriaMet,
							Status: api.ConditionTrue,
						},
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
		},
		"invalid addition of SuccessCriteriaMet for Job with Failed": {
			enableJobSuccessPolicy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobFailed,
						Status: api.ConditionTrue,
					}},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobFailed,
							Status: api.ConditionTrue,
						},
						{
							Type:   batch.JobSuccessCriteriaMet,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"invalid addition of Failed for Job with SuccessCriteriaMet": {
			enableJobSuccessPolicy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobSuccessCriteriaMet,
						Status: api.ConditionTrue,
					}},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobSuccessCriteriaMet,
							Status: api.ConditionTrue,
						},
						{
							Type:   batch.JobFailed,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"invalid addition of SuccessCriteriaMet for Job with FailureTarget": {
			enableJobSuccessPolicy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobFailureTarget,
						Status: api.ConditionTrue,
					}},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobFailureTarget,
							Status: api.ConditionTrue,
						},
						{
							Type:   batch.JobSuccessCriteriaMet,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"invalid addition of FailureTarget for Job with SuccessCriteriaMet": {
			enableJobSuccessPolicy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobSuccessCriteriaMet,
						Status: api.ConditionTrue,
					}},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobSuccessCriteriaMet,
							Status: api.ConditionTrue,
						},
						{
							Type:   batch.JobFailureTarget,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"invalid addition of SuccessCriteriaMet for Job with Complete": {
			enableJobSuccessPolicy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobComplete,
						Status: api.ConditionTrue,
					}},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
						{
							Type:   batch.JobSuccessCriteriaMet,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"valid addition of Complete for Job with SuccessCriteriaMet": {
			enableJobSuccessPolicy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobSuccessCriteriaMet,
						Status: api.ConditionTrue,
					}},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobSuccessCriteriaMet,
							Status: api.ConditionTrue,
						},
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
		},
		"invalid addition of SuccessCriteriaMet for Job without SuccessPolicy": {
			enableJobSuccessPolicy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobSuccessCriteriaMet,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"invalid addition of Complete for Job with SuccessPolicy unless SuccessCriteriaMet": {
			enableJobSuccessPolicy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{
						{
							Type:   batch.JobComplete,
							Status: api.ConditionTrue,
						},
					},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"invalid disabling of SuccessCriteriaMet for Job": {
			enableJobSuccessPolicy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobSuccessCriteriaMet,
						Status: api.ConditionTrue,
					}},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobComplete,
						Status: api.ConditionFalse,
					}},
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"invalid removing of SuccessCriteriaMet for Job": {
			enableJobSuccessPolicy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobSuccessCriteriaMet,
						Status: api.ConditionTrue,
					}},
				},
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Spec: batch.JobSpec{
					CompletionMode: completionModePtr(batch.IndexedCompletion),
					Completions:    ptr.To[int32](10),
					SuccessPolicy:  validSuccessPolicy,
				},
			},
			wantErrs: field.ErrorList{
				{Type: field.ErrorTypeInvalid, Field: "status.conditions"},
			},
		},
		"valid addition of SuccessCriteriaMet when JobManagedBy is enabled": {
			enableJobManagedBy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobSuccessCriteriaMet,
						Status: api.ConditionTrue,
					}},
				},
			},
		},
		"valid addition of SuccessCriteriaMet when JobPodReplacementPolicy is enabled": {
			enableJobPodReplacementPolicy: true,
			job: &batch.Job{
				ObjectMeta: validObjectMeta,
			},
			newJob: &batch.Job{
				ObjectMeta: validObjectMeta,
				Status: batch.JobStatus{
					Conditions: []batch.JobCondition{{
						Type:   batch.JobSuccessCriteriaMet,
						Status: api.ConditionTrue,
					}},
				},
			},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobManagedBy, tc.enableJobManagedBy)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobSuccessPolicy, tc.enableJobSuccessPolicy)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.JobPodReplacementPolicy, tc.enableJobPodReplacementPolicy)

			errs := StatusStrategy.ValidateUpdate(ctx, tc.newJob, tc.job)
			if diff := cmp.Diff(tc.wantErrs, errs, ignoreErrValueDetail); diff != "" {
				t.Errorf("Unexpected errors (-want,+got):\n%s", diff)
			}
			if tc.wantJob != nil {
				if diff := cmp.Diff(tc.wantJob, tc.newJob); diff != "" {
					t.Errorf("Unexpected job (-want,+got):\n%s", diff)
				}
			}
		})
	}
}

func TestJobStrategy_GetAttrs(t *testing.T) {
	validSelector := &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
	validPodTemplateSpec := api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}

	cases := map[string]struct {
		job          *batch.Job
		wantErr      string
		nonJobObject *api.Pod
	}{
		"valid job with no labels": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},
		},
		"valid job with a label": {
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "myjob",
					Namespace:       metav1.NamespaceDefault,
					ResourceVersion: "0",
					Labels:          map[string]string{"a": "b"},
				},
				Spec: batch.JobSpec{
					Selector:       validSelector,
					Template:       validPodTemplateSpec,
					ManualSelector: ptr.To(true),
					Parallelism:    ptr.To[int32](1),
				},
			},
		},
		"pod instead": {
			job:          nil,
			nonJobObject: &api.Pod{},
			wantErr:      "given object is not a job.",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if tc.job == nil {
				_, _, err := GetAttrs(tc.nonJobObject)
				if diff := cmp.Diff(tc.wantErr, err.Error()); diff != "" {
					t.Errorf("Unexpected errors (-want,+got):\n%s", diff)
				}
			} else {
				gotLabels, _, err := GetAttrs(tc.job)
				if err != nil {
					t.Errorf("Error %s supposed to be nil", err.Error())
				}
				if diff := cmp.Diff(labels.Set(tc.job.ObjectMeta.Labels), gotLabels); diff != "" {
					t.Errorf("Unexpected attrs (-want,+got):\n%s", diff)
				}
			}
		})
	}
}

func TestJobToSelectiableFields(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		"batch/v1",
		"Job",
		JobToSelectableFields(&batch.Job{}),
		nil,
	)
}

func completionModePtr(m batch.CompletionMode) *batch.CompletionMode {
	return &m
}

func podReplacementPolicy(m batch.PodReplacementPolicy) *batch.PodReplacementPolicy {
	return &m
}

func getValidObjectMeta(generation int64) metav1.ObjectMeta {
	return getValidObjectMetaWithAnnotations(generation, nil)
}

func getValidUID() types.UID {
	return "1a2b3c4d5e6f7g8h9i0k"
}

func getValidObjectMetaWithAnnotations(generation int64, annotations map[string]string) metav1.ObjectMeta {
	return metav1.ObjectMeta{
		Name:        "myjob",
		UID:         getValidUID(),
		Namespace:   metav1.NamespaceDefault,
		Generation:  generation,
		Annotations: annotations,
	}
}

func getValidLabelSelector() *metav1.LabelSelector {
	return &metav1.LabelSelector{
		MatchLabels: map[string]string{"a": "b"},
	}
}

func getValidBatchLabels() map[string]string {
	theUID := getValidUID()
	return map[string]string{batch.LegacyJobNameLabel: "myjob", batch.JobNameLabel: "myjob", batch.LegacyControllerUidLabel: string(theUID), batch.ControllerUidLabel: string(theUID)}
}

func getValidBatchLabelsWithNonBatch() map[string]string {
	theUID := getValidUID()
	return map[string]string{"a": "b", batch.LegacyJobNameLabel: "myjob", batch.JobNameLabel: "myjob", batch.LegacyControllerUidLabel: string(theUID), batch.ControllerUidLabel: string(theUID)}
}

func getValidPodTemplateSpecForSelector(validSelector *metav1.LabelSelector) api.PodTemplateSpec {
	return api.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: validSelector.MatchLabels,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyOnFailure,
			DNSPolicy:     api.DNSClusterFirst,
			Containers:    []api.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
	}
}
