/*
Copyright 2020 The Kubernetes Authors.

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

package denyserviceexternalips

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/apis/core"
)

func makeSvc(externalIPs ...string) *core.Service {
	svc := &core.Service{}
	svc.Namespace = "test-ns"
	svc.Name = "test-svc"
	svc.Spec.ExternalIPs = externalIPs
	return svc
}

func TestAdmission(t *testing.T) {
	testCases := []struct {
		name   string
		newSvc *core.Service
		oldSvc *core.Service
		fail   bool
	}{{
		name:   "create: without externalIPs",
		newSvc: makeSvc(),
	}, {
		name:   "create: with externalIPs",
		newSvc: makeSvc("1.1.1.1"),
		fail:   true,
	}, {
		name:   "update: same externalIPs",
		newSvc: makeSvc("1.1.1.1", "2.2.2.2"),
		oldSvc: makeSvc("1.1.1.1", "2.2.2.2"),
	}, {
		name:   "update: reorder externalIPs",
		newSvc: makeSvc("1.1.1.1", "2.2.2.2"),
		oldSvc: makeSvc("2.2.2.2", "1.1.1.1"),
	}, {
		name:   "update: change externalIPs",
		newSvc: makeSvc("1.1.1.1", "2.2.2.2"),
		oldSvc: makeSvc("1.1.1.1", "3.3.3.3"),
		fail:   true,
	}, {
		name:   "update: add externalIPs",
		newSvc: makeSvc("1.1.1.1", "2.2.2.2"),
		oldSvc: makeSvc("1.1.1.1"),
		fail:   true,
	}, {
		name:   "update: erase externalIPs",
		newSvc: makeSvc(),
		oldSvc: makeSvc("1.1.1.1", "2.2.2.2"),
	}, {
		name:   "update: reduce externalIPs from back",
		newSvc: makeSvc("1.1.1.1"),
		oldSvc: makeSvc("1.1.1.1", "2.2.2.2"),
	}, {
		name:   "update: reduce externalIPs from front",
		newSvc: makeSvc("2.2.2.2"),
		oldSvc: makeSvc("1.1.1.1", "2.2.2.2"),
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctrl := newPlugin()

			var op admission.Operation
			var opt runtime.Object
			if tc.oldSvc == nil {
				op = admission.Create
				opt = &metav1.CreateOptions{}
			} else {
				op = admission.Update
				opt = &metav1.UpdateOptions{}
			}

			attrs := admission.NewAttributesRecord(
				tc.newSvc, // new object
				tc.oldSvc, // old object
				core.Kind("Service").WithVersion("version"),
				tc.newSvc.Namespace,
				tc.newSvc.Name,
				corev1.Resource("services").WithVersion("version"),
				"", // subresource
				op,
				opt,
				false, // dryRun
				nil,   // userInfo
			)

			err := ctrl.Validate(context.TODO(), attrs, nil)
			if err != nil && !tc.fail {
				t.Errorf("Unexpected failure: %v", err)
			}
			if err == nil && tc.fail {
				t.Errorf("Unexpected success")
			}
		})
	}
}
