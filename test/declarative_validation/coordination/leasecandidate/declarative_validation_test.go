/*
Copyright The Kubernetes Authors.

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

package leasecandidate

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	coordination "k8s.io/kubernetes/pkg/apis/coordination"
	registry "k8s.io/kubernetes/pkg/registry/coordination/leasecandidate"
	"k8s.io/kubernetes/test/declarative_validation/meta"
)

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidate(t, apiVersion)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			testDeclarativeValidateUpdate(t, apiVersion)
		})
	}
}

func testDeclarativeValidate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "coordination.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "leasecandidates",
		IsResourceRequest: true,
		Verb:              "create",
	}), metav1.NamespaceDefault)

	obj := mkValidLeaseCandidate()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy)
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "coordination.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "leasecandidates",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	}), metav1.NamespaceDefault)

	updateObj := mkValidLeaseCandidate()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &updateObj, registry.Strategy)
}

func mkValidLeaseCandidate() coordination.LeaseCandidate {
	return coordination.LeaseCandidate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-obj",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: coordination.LeaseCandidateSpec{
			LeaseName:        "lease",
			BinaryVersion:    "1.0.0",
			EmulationVersion: "1.0.0",
			Strategy:         coordination.OldestEmulationVersion,
		},
	}
}
