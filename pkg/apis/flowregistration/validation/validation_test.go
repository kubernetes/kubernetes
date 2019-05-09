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
	"testing"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/flowregistration"
)

// TODO(aaron-prindle) finish validation_test.go
func TestValidateFlowSchema(t *testing.T) {
	testCases := []struct {
		name   string
		conf   flowregistration.FlowSchema
		numErr int
	}{
		{
			name: "should pass full config",
			conf: flowregistration.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-top",
				},
				Spec: flowregistration.FlowSchemaSpec{
					RequestPriority: flowregistration.RequestPriority{
						Name: "system-top",
					},
					FlowDistinguisher: flowregistration.FlowDistinguisher{
						Source: "user",
					},
					Match: []*flowregistration.Match{
						&flowregistration.Match{
							And: &flowregistration.And{
								Equals: flowregistration.Equals{
									Field: "user",
									Value: "u1",
								},
							},
						},
					},
				},
			},
			numErr: 0,
		},
		{
			name: "should no match criteria",
			conf: flowregistration.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: "system-top",
				},
				Spec: flowregistration.FlowSchemaSpec{
					RequestPriority: flowregistration.RequestPriority{
						Name: "system-top",
					},
					FlowDistinguisher: flowregistration.FlowDistinguisher{
						Source: "user",
					},
					Match: []*flowregistration.Match{},
				},
			},
			numErr: 1,
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			errs := ValidateFlowSchema(&test.conf)
			require.Len(t, errs, test.numErr)
		})
	}
}

func TestRequestPriority(t *testing.T) {
	successCases := []flowregistration.RequestPriority{}
	successCases = append(successCases, flowregistration.RequestPriority{ // Policy with omitStages and level
		Name: "system-top",
	})

	for i, rq := range successCases {
		if errs := ValidateRequestPriority(rq, field.NewPath("requestPriority")); len(errs) != 0 {
			t.Errorf("[%d] Expected policy %#v to be valid: %v", i, rq, errs)
		}
	}

	errorCases := []flowregistration.RequestPriority{}
	errorCases = append(successCases, flowregistration.RequestPriority{ // Policy with omitStages and level
	})

	for i, policy := range errorCases {
		if errs := ValidateRequestPriority(policy, field.NewPath("policy")); len(errs) == 0 {
			t.Errorf("[%d] Expected policy %#v to be invalid!", i, policy)
		}
	}
}
