/*
Copyright 2017 The Kubernetes Authors.

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

package authorizationconditionsreview

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/admission"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	autoscalingapi "k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

type fakeAuthorizer struct {
	evaluateDecision authorizer.Decision
	evaluateErr      error

	gotDecision authorizer.Decision
	gotData     authorizer.ConditionData
}

func (f *fakeAuthorizer) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, error) {
	return authorizer.DecisionNoOpinion(), nil
}

func (f *fakeAuthorizer) EvaluateConditions(ctx context.Context, decision authorizer.Decision, data authorizer.ConditionData) (authorizer.Decision, error) {
	f.gotDecision = decision
	f.gotData = data
	return f.evaluateDecision, f.evaluateErr
}

type expectedConditionData struct {
	writeOperation string
	writeObject    runtime.Object
	writeOldObject runtime.Object
}

func TestCreate(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)
	podJSON := []byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {"name": "test-pod", "namespace": "default"},
		"spec": {"containers": [{"name": "nginx", "image": "nginx:latest"}]}
	}`)

	newPodJSON := []byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {"name": "new-pod", "namespace": "default"},
		"spec": {"containers": [{"name": "nginx", "image": "nginx:latest"}]}
	}`)

	oldPodJSON := []byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {"name": "old-pod", "namespace": "default"},
		"spec": {"containers": [{"name": "nginx", "image": "nginx:latest"}]}
	}`)

	hpaJSON := []byte(`{
		"apiVersion": "autoscaling/v1",
		"kind": "HorizontalPodAutoscaler",
		"metadata": {"name": "test-hpa", "namespace": "default"},
		"spec": {"maxReplicas": 10, "targetCPUUtilizationPercentage": 80}
	}`)

	unregisteredJSON := []byte(`{
		"apiVersion": "example.com/v1",
		"kind": "Foo",
		"metadata": {"name": "my-foo", "namespace": "bar"},
		"spec": {"field1": "value1"}
	}`)

	expectedPod := func(name, namespace string) *api.Pod {
		return &api.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
			Spec: api.PodSpec{
				Containers: []api.Container{{
					Name:                     "nginx",
					Image:                    "nginx:latest",
					TerminationMessagePath:   "/dev/termination-log",
					TerminationMessagePolicy: api.TerminationMessageReadFile,
					ImagePullPolicy:          api.PullAlways,
				}},
				RestartPolicy:                 api.RestartPolicyAlways,
				TerminationGracePeriodSeconds: ptr.To(int64(30)),
				DNSPolicy:                     api.DNSClusterFirst,
				SecurityContext:               &api.PodSecurityContext{},
				SchedulerName:                 "default-scheduler",
				EnableServiceLinks:            ptr.To(true),
			},
		}
	}

	serializedConditionSet := func(conditionsType, id, cond string, effect authorizationapi.SubjectAccessReviewConditionEffect) authorizationapi.SubjectAccessReviewAuthorizationDecision {
		return authorizationapi.SubjectAccessReviewAuthorizationDecision{
			ConditionsType: conditionsType,
			Conditions: []authorizationapi.SubjectAccessReviewCondition{
				{
					ID:        id,
					Condition: cond,
					Effect:    effect,
				},
			},
		}
	}

	sampleConditionSet := serializedConditionSet("foo", "test-cond-id", "object.metadata.labels.foo == 'bar'", authorizationapi.SubjectAccessReviewConditionEffectAllow)

	tests := []struct {
		name             string
		input            runtime.Object
		evaluateDecision authorizer.Decision
		evaluateErr      error

		expectedErr           string
		expectedConditionData *expectedConditionData
		expectedResponse      *authorizationapi.AuthorizationConditionsResponse
	}{
		{
			name:        "wrong object type",
			input:       &api.Pod{},
			expectedErr: "not a AuthorizationConditionsReview",
		},
		{
			name:        "nil request",
			input:       &authorizationapi.AuthorizationConditionsReview{},
			expectedErr: "must be set",
		},
		{
			name: "nil write request",
			input: &authorizationapi.AuthorizationConditionsReview{
				Request: &authorizationapi.AuthorizationConditionsRequest{
					Decision: sampleConditionSet,
				},
			},
			expectedErr: "at least one type of conditions data",
		},
		{
			name: "mutually exclusive allowed and denied",
			input: &authorizationapi.AuthorizationConditionsReview{
				Request: &authorizationapi.AuthorizationConditionsRequest{
					Decision: authorizationapi.SubjectAccessReviewAuthorizationDecision{
						Allowed: true,
						Denied:  true,
					},
					WriteRequest: &authorizationapi.AuthorizationConditionsWriteRequest{
						Operation: admission.Create,
					},
				},
			},
			expectedErr: "mutually exclusive",
		},
		{
			name: "invalid JSON in object",
			input: &authorizationapi.AuthorizationConditionsReview{
				Request: &authorizationapi.AuthorizationConditionsRequest{
					Decision: sampleConditionSet,
					WriteRequest: &authorizationapi.AuthorizationConditionsWriteRequest{
						Operation: admission.Create,
						Object:    runtime.RawExtension{Raw: []byte(`{not valid`)},
					},
				},
			},
			expectedErr: "invalid character",
		},
		{
			name: "create pod, evaluate allows",
			input: &authorizationapi.AuthorizationConditionsReview{
				Request: &authorizationapi.AuthorizationConditionsRequest{
					Decision: sampleConditionSet,
					WriteRequest: &authorizationapi.AuthorizationConditionsWriteRequest{
						Operation: admission.Create,
						Object:    runtime.RawExtension{Raw: podJSON},
					},
				},
			},
			evaluateDecision: authorizer.DecisionAllow("allowed"),
			expectedConditionData: &expectedConditionData{
				writeOperation: "CREATE",
				writeObject:    expectedPod("test-pod", "default"),
			},
			expectedResponse: &authorizationapi.AuthorizationConditionsResponse{
				SubjectAccessReviewAuthorizationDecision: authorizationapi.SubjectAccessReviewAuthorizationDecision{
					Allowed: true,
					Reason:  "allowed",
				},
			},
		},
		{
			name: "create HPA with v1 defaulting and conversion",
			input: &authorizationapi.AuthorizationConditionsReview{
				Request: &authorizationapi.AuthorizationConditionsRequest{
					Decision: sampleConditionSet,
					WriteRequest: &authorizationapi.AuthorizationConditionsWriteRequest{
						Operation: admission.Create,
						Object:    runtime.RawExtension{Raw: hpaJSON},
					},
				},
			},
			evaluateDecision: authorizer.DecisionAllow("ok"),
			expectedConditionData: &expectedConditionData{
				writeOperation: "CREATE",
				writeObject: &autoscalingapi.HorizontalPodAutoscaler{
					ObjectMeta: metav1.ObjectMeta{Name: "test-hpa", Namespace: "default"},
					Spec: autoscalingapi.HorizontalPodAutoscalerSpec{
						MinReplicas: ptr.To(int32(1)), // from v1 defaults
						MaxReplicas: 10,
						// from v1 -> internal conversion
						Metrics: []autoscalingapi.MetricSpec{{
							Type: autoscalingapi.ResourceMetricSourceType,
							Resource: &autoscalingapi.ResourceMetricSource{
								Name: api.ResourceCPU,
								Target: autoscalingapi.MetricTarget{
									Type:               autoscalingapi.UtilizationMetricType,
									AverageUtilization: ptr.To(int32(80)),
								},
							},
						}},
					},
				},
			},
			expectedResponse: &authorizationapi.AuthorizationConditionsResponse{
				SubjectAccessReviewAuthorizationDecision: authorizationapi.SubjectAccessReviewAuthorizationDecision{
					Allowed: true,
					Reason:  "ok",
				},
			},
		},
		{
			name: "unregistered type falls back to unstructured",
			input: &authorizationapi.AuthorizationConditionsReview{
				Request: &authorizationapi.AuthorizationConditionsRequest{
					Decision: sampleConditionSet,
					WriteRequest: &authorizationapi.AuthorizationConditionsWriteRequest{
						Operation: admission.Create,
						Object:    runtime.RawExtension{Raw: unregisteredJSON},
					},
				},
			},
			evaluateDecision: authorizer.DecisionDeny("not ok"),
			expectedConditionData: &expectedConditionData{
				writeOperation: "CREATE",
				writeObject: &unstructured.Unstructured{
					Object: map[string]any{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]any{"name": "my-foo", "namespace": "bar"},
						"spec":       map[string]any{"field1": "value1"},
					},
				},
			},
			expectedResponse: &authorizationapi.AuthorizationConditionsResponse{
				SubjectAccessReviewAuthorizationDecision: authorizationapi.SubjectAccessReviewAuthorizationDecision{
					Denied: true,
					Reason: "not ok",
				},
			},
		},
		{
			name: "update pod with object and old object, evaluate denies",
			input: &authorizationapi.AuthorizationConditionsReview{
				Request: &authorizationapi.AuthorizationConditionsRequest{
					Decision: sampleConditionSet,
					WriteRequest: &authorizationapi.AuthorizationConditionsWriteRequest{
						Operation: admission.Update,
						Object:    runtime.RawExtension{Raw: newPodJSON},
						OldObject: runtime.RawExtension{Raw: oldPodJSON},
					},
				},
			},
			evaluateDecision: authorizer.DecisionDeny("denied"),
			expectedConditionData: &expectedConditionData{
				writeOperation: "UPDATE",
				writeObject:    expectedPod("new-pod", "default"),
				writeOldObject: expectedPod("old-pod", "default"),
			},
			expectedResponse: &authorizationapi.AuthorizationConditionsResponse{
				SubjectAccessReviewAuthorizationDecision: authorizationapi.SubjectAccessReviewAuthorizationDecision{
					Denied: true,
					Reason: "denied",
				},
			},
		},
		{
			name: "evaluate returns no opinion from a union authorizer",
			input: &authorizationapi.AuthorizationConditionsReview{
				Request: &authorizationapi.AuthorizationConditionsRequest{
					Decision: authorizationapi.SubjectAccessReviewAuthorizationDecision{
						ConditionalDecisionChain: []authorizationapi.SubjectAccessReviewAuthorizationDecision{
							sampleConditionSet,
							sampleConditionSet,
						},
					},
					WriteRequest: &authorizationapi.AuthorizationConditionsWriteRequest{
						Operation: admission.Create,
						Object:    runtime.RawExtension{Raw: podJSON},
					},
				},
			},
			evaluateDecision: authorizer.DecisionNoOpinion("unsure"),
			expectedConditionData: &expectedConditionData{
				writeOperation: "CREATE",
				writeObject:    expectedPod("test-pod", "default"),
			},
			expectedResponse: &authorizationapi.AuthorizationConditionsResponse{
				SubjectAccessReviewAuthorizationDecision: authorizationapi.SubjectAccessReviewAuthorizationDecision{
					Reason: "unsure",
				},
			},
		},
		{
			name: "evaluate returns error",
			input: &authorizationapi.AuthorizationConditionsReview{
				Request: &authorizationapi.AuthorizationConditionsRequest{
					Decision: sampleConditionSet,
					WriteRequest: &authorizationapi.AuthorizationConditionsWriteRequest{
						Operation: admission.Create,
						Object:    runtime.RawExtension{Raw: podJSON},
					},
				},
			},
			evaluateErr: errors.New("evaluate failed"),
			expectedConditionData: &expectedConditionData{
				writeOperation: "CREATE",
				writeObject:    expectedPod("test-pod", "default"),
			},
			expectedErr: "evaluate failed",
		},
		{
			name: "failed to parse conditiontype",
			input: &authorizationapi.AuthorizationConditionsReview{
				Request: &authorizationapi.AuthorizationConditionsRequest{
					Decision: serializedConditionSet("has whitespace", "test-cond-id", "object.metadata.labels.foo == 'bar'", authorizationapi.SubjectAccessReviewConditionEffectAllow),
					WriteRequest: &authorizationapi.AuthorizationConditionsWriteRequest{
						Operation: admission.Create,
						Object:    runtime.RawExtension{Raw: podJSON},
					},
				},
			},
			expectedErr: "invalid condition type",
		},
		{
			name: "failed to parse conditiontype",
			input: &authorizationapi.AuthorizationConditionsReview{
				Request: &authorizationapi.AuthorizationConditionsRequest{
					Decision: serializedConditionSet("test-cond-type", "invalid id", "object.metadata.labels.foo == 'bar'", authorizationapi.SubjectAccessReviewConditionEffectAllow),
					WriteRequest: &authorizationapi.AuthorizationConditionsWriteRequest{
						Operation: admission.Create,
						Object:    runtime.RawExtension{Raw: podJSON},
					},
				},
			},
			expectedErr: "invalid condition ID",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			auth := &fakeAuthorizer{
				evaluateDecision: tc.evaluateDecision,
				evaluateErr:      tc.evaluateErr,
			}
			r, err := NewREST(auth, legacyscheme.Codecs)
			if err != nil {
				t.Fatalf("NewREST failed: %v", err)
			}

			result, err := r.Create(t.Context(), tc.input, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})

			// Check condition data passed to EvaluateConditions (before error check,
			// since EvaluateConditions may have been called even when it returns an error).
			if tc.expectedConditionData != nil {
				if auth.gotData == nil {
					t.Fatal("expected EvaluateConditions to be called, but it was not")
				}
				wr := auth.gotData.WriteRequest()
				if wr == nil {
					t.Fatal("expected non-nil WriteRequest in condition data")
				}
				if wr.GetOperation() != tc.expectedConditionData.writeOperation {
					t.Errorf("condition data operation: got %q, want %q", wr.GetOperation(), tc.expectedConditionData.writeOperation)
				}
				if diff := cmp.Diff(tc.expectedConditionData.writeObject, wr.GetObject()); diff != "" {
					t.Errorf("condition data object mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(tc.expectedConditionData.writeOldObject, wr.GetOldObject()); diff != "" {
					t.Errorf("condition data old object mismatch (-want +got):\n%s", diff)
				}
			}

			if tc.expectedErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.expectedErr)
				}
				if !strings.Contains(err.Error(), tc.expectedErr) {
					t.Errorf("error mismatch: got %q, want containing %q", err.Error(), tc.expectedErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			review, ok := result.(*authorizationapi.AuthorizationConditionsReview)
			if !ok {
				t.Fatalf("expected *AuthorizationConditionsReview, got %T", result)
			}
			if diff := cmp.Diff(tc.expectedResponse, review.Response); diff != "" {
				t.Errorf("response mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
