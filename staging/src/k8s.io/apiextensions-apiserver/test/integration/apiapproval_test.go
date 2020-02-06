/*
Copyright 2019 The Kubernetes Authors.

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

package integration

import (
	"testing"
	"time"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
)

func TestAPIApproval(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinition := fixtures.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}
	if noxuAPIApproved := findCRDCondition(noxuDefinition, apiextensionsv1beta1.KubernetesAPIApprovalPolicyConformant); noxuAPIApproved != nil {
		t.Fatal(noxuAPIApproved)
	}

	newSigKubeAPIFn := func(resource, approvalAnnotation string) *apiextensionsv1beta1.CustomResourceDefinition {
		return &apiextensionsv1beta1.CustomResourceDefinition{
			ObjectMeta: metav1.ObjectMeta{Name: resource + ".sigs.k8s.io", Annotations: map[string]string{apiextensionsv1beta1.KubeAPIApprovedAnnotation: approvalAnnotation}},
			Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
				Group:   "sigs.k8s.io",
				Version: "v1beta1",
				Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
					Plural:   resource,
					Singular: resource + "singular",
					Kind:     resource + "Kind",
					ListKind: resource + "List",
				},
				Scope: apiextensionsv1beta1.NamespaceScoped,
			},
		}
	}
	// the unit tests cover all variations. We just need to be sure that we see the code being called
	approvedKubeAPI := newSigKubeAPIFn("approved", "https://github.com/kubernetes/kubernetes/pull/79724")
	approvedKubeAPI, err = fixtures.CreateNewCustomResourceDefinition(approvedKubeAPI, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}
	err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		approvedKubeAPI, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(approvedKubeAPI.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if approvedKubeAPIApproved := findCRDCondition(approvedKubeAPI, apiextensionsv1beta1.KubernetesAPIApprovalPolicyConformant); approvedKubeAPIApproved == nil || approvedKubeAPIApproved.Status != apiextensionsv1beta1.ConditionTrue {
			t.Log(approvedKubeAPIApproved)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatal(err)
	}

}
