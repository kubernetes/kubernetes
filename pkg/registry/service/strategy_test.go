/*
Copyright 2014 The Kubernetes Authors.

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

package service

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func TestExportService(t *testing.T) {
	tests := []struct {
		objIn     runtime.Object
		objOut    runtime.Object
		exact     bool
		expectErr bool
	}{
		{
			objIn: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
				Status: api.ServiceStatus{
					LoadBalancer: api.LoadBalancerStatus{
						Ingress: []api.LoadBalancerIngress{
							{IP: "1.2.3.4"},
						},
					},
				},
			},
			objOut: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
			},
			exact: true,
		},
		{
			objIn: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
				Spec: api.ServiceSpec{
					ClusterIP: "10.0.0.1",
				},
				Status: api.ServiceStatus{
					LoadBalancer: api.LoadBalancerStatus{
						Ingress: []api.LoadBalancerIngress{
							{IP: "1.2.3.4"},
						},
					},
				},
			},
			objOut: &api.Service{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
				Spec: api.ServiceSpec{
					ClusterIP: "<unknown>",
				},
			},
		},
		{
			objIn:     &api.Pod{},
			expectErr: true,
		},
	}

	for _, test := range tests {
		err := Strategy.Export(api.NewContext(), test.objIn, test.exact)
		if err != nil {
			if !test.expectErr {
				t.Errorf("unexpected error: %v", err)
			}
			continue
		}
		if test.expectErr {
			t.Error("unexpected non-error")
			continue
		}
		if !reflect.DeepEqual(test.objIn, test.objOut) {
			t.Errorf("expected:\n%v\nsaw:\n%v\n", test.objOut, test.objIn)
		}
	}
}

func TestCheckGeneratedNameError(t *testing.T) {
	expect := errors.NewNotFound(api.Resource("foos"), "bar")
	if err := rest.CheckGeneratedNameError(Strategy, expect, &api.Pod{}); err != expect {
		t.Errorf("NotFoundError should be ignored: %v", err)
	}

	expect = errors.NewAlreadyExists(api.Resource("foos"), "bar")
	if err := rest.CheckGeneratedNameError(Strategy, expect, &api.Pod{}); err != expect {
		t.Errorf("AlreadyExists should be returned when no GenerateName field: %v", err)
	}

	expect = errors.NewAlreadyExists(api.Resource("foos"), "bar")
	if err := rest.CheckGeneratedNameError(Strategy, expect, &api.Pod{ObjectMeta: api.ObjectMeta{GenerateName: "foo"}}); err == nil || !errors.IsServerTimeout(err) {
		t.Errorf("expected try again later error: %v", err)
	}
}

func makeValidService() api.Service {
	return api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:            "valid",
			Namespace:       "default",
			Labels:          map[string]string{},
			Annotations:     map[string]string{},
			ResourceVersion: "1",
		},
		Spec: api.ServiceSpec{
			Selector:        map[string]string{"key": "val"},
			SessionAffinity: "None",
			Type:            api.ServiceTypeClusterIP,
			Ports:           []api.ServicePort{{Name: "p", Protocol: "TCP", Port: 8675, TargetPort: intstr.FromInt(8675)}},
		},
	}
}

// TODO: This should be done on types that are not part of our API
func TestBeforeUpdate(t *testing.T) {
	testCases := []struct {
		name      string
		tweakSvc  func(oldSvc, newSvc *api.Service) // given basic valid services, each test case can customize them
		expectErr bool
	}{
		{
			name: "no change",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				// nothing
			},
			expectErr: false,
		},
		{
			name: "change port",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Spec.Ports[0].Port++
			},
			expectErr: false,
		},
		{
			name: "bad namespace",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Namespace = "#$%%invalid"
			},
			expectErr: true,
		},
		{
			name: "change name",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Name += "2"
			},
			expectErr: true,
		},
		{
			name: "change ClusterIP",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				oldSvc.Spec.ClusterIP = "1.2.3.4"
				newSvc.Spec.ClusterIP = "4.3.2.1"
			},
			expectErr: true,
		},
		{
			name: "change selectpor",
			tweakSvc: func(oldSvc, newSvc *api.Service) {
				newSvc.Spec.Selector = map[string]string{"newkey": "newvalue"}
			},
			expectErr: false,
		},
	}

	for _, tc := range testCases {
		oldSvc := makeValidService()
		newSvc := makeValidService()
		tc.tweakSvc(&oldSvc, &newSvc)
		ctx := api.NewDefaultContext()
		err := rest.BeforeUpdate(Strategy, ctx, runtime.Object(&oldSvc), runtime.Object(&newSvc))
		if tc.expectErr && err == nil {
			t.Errorf("unexpected non-error for %q", tc.name)
		}
		if !tc.expectErr && err != nil {
			t.Errorf("unexpected error for %q: %v", tc.name, err)
		}
	}
}

func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		testapi.Default.GroupVersion().String(),
		"Service",
		labels.Set(ServiceToSelectableFields(&api.Service{})),
		nil,
	)
}

func TestServiceStatusStrategy(t *testing.T) {
	ctx := api.NewDefaultContext()
	if !StatusStrategy.NamespaceScoped() {
		t.Errorf("Service must be namespace scoped")
	}
	oldService := makeValidService()
	newService := makeValidService()
	oldService.ResourceVersion = "4"
	newService.ResourceVersion = "4"
	newService.Spec.SessionAffinity = "ClientIP"
	newService.Status = api.ServiceStatus{
		LoadBalancer: api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "127.0.0.2"},
			},
		},
	}
	StatusStrategy.PrepareForUpdate(ctx, &newService, &oldService)
	if newService.Status.LoadBalancer.Ingress[0].IP != "127.0.0.2" {
		t.Errorf("Service status updates should allow change of status fields")
	}
	if newService.Spec.SessionAffinity != "None" {
		t.Errorf("PrepareForUpdate should have preserved old spec")
	}
	errs := StatusStrategy.ValidateUpdate(ctx, &newService, &oldService)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}
}
