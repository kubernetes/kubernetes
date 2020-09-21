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

package apimachinery

import (
	"context"
	"net/http"

	"github.com/onsi/ginkgo"
	"k8s.io/client-go/rest"

	flowcontrolv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = SIGDescribe("[Feature:APIPriorityAndFairness] response header should present", func() {
	f := framework.NewDefaultFramework("flowschemas")

	ginkgo.It("should ensure that requests can be classified by testing flow-schemas/priority-levels", func() {
		testingFlowSchemaName := "e2e-testing-flowschema"
		testingPriorityLevelName := "e2e-testing-prioritylevel"
		matchingUsername := "noxu"
		nonMatchingUsername := "foo"

		ginkgo.By("creating a testing prioritylevel")
		createdPriorityLevel, err := f.ClientSet.FlowcontrolV1alpha1().PriorityLevelConfigurations().Create(
			context.TODO(),
			&flowcontrolv1alpha1.PriorityLevelConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name: testingPriorityLevelName,
				},
				Spec: flowcontrolv1alpha1.PriorityLevelConfigurationSpec{
					Type: flowcontrolv1alpha1.PriorityLevelEnablementLimited,
					Limited: &flowcontrolv1alpha1.LimitedPriorityLevelConfiguration{
						AssuredConcurrencyShares: 1, // will have at minimum 1 concurrency share
						LimitResponse: flowcontrolv1alpha1.LimitResponse{
							Type: flowcontrolv1alpha1.LimitResponseTypeReject,
						},
					},
				},
			},
			metav1.CreateOptions{})
		framework.ExpectNoError(err)

		defer func() {
			// clean-ups
			err := f.ClientSet.FlowcontrolV1alpha1().PriorityLevelConfigurations().Delete(context.TODO(), testingPriorityLevelName, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
			err = f.ClientSet.FlowcontrolV1alpha1().FlowSchemas().Delete(context.TODO(), testingFlowSchemaName, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		}()

		ginkgo.By("creating a testing flowschema")
		createdFlowSchema, err := f.ClientSet.FlowcontrolV1alpha1().FlowSchemas().Create(
			context.TODO(),
			&flowcontrolv1alpha1.FlowSchema{
				ObjectMeta: metav1.ObjectMeta{
					Name: testingFlowSchemaName,
				},
				Spec: flowcontrolv1alpha1.FlowSchemaSpec{
					MatchingPrecedence: 1000, // a rather higher precedence to ensure it make effect
					PriorityLevelConfiguration: flowcontrolv1alpha1.PriorityLevelConfigurationReference{
						Name: testingPriorityLevelName,
					},
					DistinguisherMethod: &flowcontrolv1alpha1.FlowDistinguisherMethod{
						Type: flowcontrolv1alpha1.FlowDistinguisherMethodByUserType,
					},
					Rules: []flowcontrolv1alpha1.PolicyRulesWithSubjects{
						{
							Subjects: []flowcontrolv1alpha1.Subject{
								{
									Kind: flowcontrolv1alpha1.SubjectKindUser,
									User: &flowcontrolv1alpha1.UserSubject{
										Name: matchingUsername,
									},
								},
							},
							NonResourceRules: []flowcontrolv1alpha1.NonResourcePolicyRule{
								{
									Verbs:           []string{flowcontrolv1alpha1.VerbAll},
									NonResourceURLs: []string{flowcontrolv1alpha1.NonResourceAll},
								},
							},
						},
					},
				},
			},
			metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("response headers should contain flow-schema/priority-level uid")

		if !testResponseHeaderMatches(f, matchingUsername, string(createdPriorityLevel.UID), string(createdFlowSchema.UID)) {
			framework.Failf("matching user doesnt received UID for the testing priority-level and flow-schema")
		}
		if testResponseHeaderMatches(f, nonMatchingUsername, string(createdPriorityLevel.UID), string(createdPriorityLevel.UID)) {
			framework.Failf("non-matching user unexpectedly received UID for the testing priority-level and flow-schema")
		}
	})

})

func testResponseHeaderMatches(f *framework.Framework, impersonatingUser, plUID, fsUID string) bool {
	config := rest.CopyConfig(f.ClientConfig())
	config.Impersonate.UserName = impersonatingUser
	roundTripper, err := rest.TransportFor(config)
	framework.ExpectNoError(err)

	req, err := http.NewRequest(http.MethodGet, f.ClientSet.CoreV1().RESTClient().Get().AbsPath("version").URL().String(), nil)
	framework.ExpectNoError(err)

	response, err := roundTripper.RoundTrip(req)
	framework.ExpectNoError(err)

	if response.Header.Get(flowcontrolv1alpha1.ResponseHeaderMatchedFlowSchemaUID) != fsUID {
		return false
	}
	if response.Header.Get(flowcontrolv1alpha1.ResponseHeaderMatchedPriorityLevelConfigurationUID) != plUID {
		return false
	}
	return true
}
