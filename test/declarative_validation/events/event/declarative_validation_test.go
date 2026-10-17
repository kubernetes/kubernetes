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

package event

import (
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	core "k8s.io/kubernetes/pkg/apis/core"
	registry "k8s.io/kubernetes/pkg/registry/core/event"
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
		APIGroup:          "events.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "events",
		IsResourceRequest: true,
		Verb:              "create",
	}), metav1.NamespaceDefault)

	t.Run("baseline", func(t *testing.T) {
		obj := mkValidEvent()
		apitesting.VerifyValidationEquivalence(t, ctx, &obj, registry.Strategy, nil)
	})

	obj := mkValidEvent()
	meta.RunObjectMetaTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())
}

func testDeclarativeValidateUpdate(t *testing.T, apiVersion string) {
	ctx := genericapirequest.WithNamespace(genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIPrefix:         "apis",
		APIGroup:          "events.k8s.io",
		APIVersion:        apiVersion,
		Resource:          "events",
		Name:              "valid-obj",
		IsResourceRequest: true,
		Verb:              "update",
	}), metav1.NamespaceDefault)

	t.Run("baseline", func(t *testing.T) {
		oldObj := mkValidEvent()
		newObj := mkValidEvent()
		newObj.EventTime = oldObj.EventTime // Make sure eventTime is identical to avoid immutability check failure
		oldObj.SetResourceVersion("1")
		newObj.SetResourceVersion("2")
		// v1/events has different validations than events.k8s.io/v1/events.
		apitesting.VerifyUpdateValidationEquivalence(t, ctx, &newObj, &oldObj, registry.Strategy, nil)
	})

	obj := mkValidEvent()
	meta.RunObjectMetaUpdateTestCases(t, ctx, &obj, registry.Strategy, meta.WithStringentFinalizerValidation())

}

func mkValidEvent() core.Event {
	return core.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-obj",
			Namespace: metav1.NamespaceDefault,
		},
		InvolvedObject: core.ObjectReference{
			Namespace: metav1.NamespaceDefault,
		},
		EventTime:           metav1.NewMicroTime(time.Now()),
		Type:                "Normal",
		ReportingController: "my-controller",
		ReportingInstance:   "my-instance",
		Action:              "my-action",
		Reason:              "my-reason",
	}
}
