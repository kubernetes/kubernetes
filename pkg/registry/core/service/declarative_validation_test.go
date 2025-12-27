/*
Copyright 2025 The Kubernetes Authors.

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
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	api "k8s.io/kubernetes/pkg/apis/core"
)

var supportedPortProtocols = sets.New(
	core.ProtocolTCP,
	core.ProtocolUDP,
	core.ProtocolSCTP)

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		input        api.Service
		expectedErrs field.ErrorList
	}{
		// baseline
		"no change": {
			input: *mkValidService(),
		},
		"invalid name": {
			input : *mkValidService(func(svc *api.Service) {
				svc.ObjectMeta.Name = "_invalid_name_"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, ""),
			},
		},
		"empty port name": {
			input: *mkValidService(func(svc *api.Service) {
				svc.Spec.Ports[0].Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec.ports[0].name"), ""),
			},
		},
		"invalid port name": {
			input: *mkValidService(func(svc *api.Service) {
				svc.Spec.Ports[0].Name = "port.one"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.ports[0].name"), nil, "").WithOrigin("format=k8s-short-name"),
			},
		},
		"duplicate ports names": {
			input: *mkValidService(func(svc *api.Service) {
				svc.Spec.Ports[0].Name = "p"
				svc.Spec.Ports[1].Name = "p"
			}),
			expectedErrs: field.ErrorList{
				field.Duplicate(field.NewPath("spec.ports[1].name"), nil),
			},
		},
		"supported protocol": {
			input: *mkValidService(func(svc *api.Service) {
				svc.Spec.Ports[0].Protocol = "TCP"
			}),
		},
		"unsupported protocol": {
			input: *mkValidService(func(svc *api.Service) {
				svc.Spec.Ports[0].Protocol = "HTTP"
			}),
			expectedErrs: field.ErrorList{
				field.NotSupported(field.NewPath("spec.ports[0].protocol"), nil, sets.List(supportedPortProtocols)),
			},
		},
		"cluster ip mismatch": {
			input: *mkValidService(func(svc *api.Service) {
				svc.Spec.ClusterIP = "1.2.3.5"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.clusterIPs"), nil, ""),
			},
		},
		"invalid cluster ip": {
			input: *mkValidService(func(svc *api.Service) {
				svc.Spec.ClusterIP = "1.2.3"
				svc.Spec.ClusterIPs[0] = "1.2.3"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.clusterIPs[0]"), nil, "").WithOrigin("format=ip-sloppy"),
			},
		},
		// TODO: Add more comprehensive tests.
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
            apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		old          api.Service
		update       api.Service
		expectedErrs field.ErrorList
	}{
		// baseline
		"no change": {
			old:    *mkValidService(),
			update: *mkValidService(),
		},
		"different cluster ip": {
			old:    *mkValidService(),
			update: *mkValidService(func (svc *api.Service)  {
				svc.Spec.ClusterIPs[0] = "1.2.3.5"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.clusterIPs"), nil, ""), 
				field.Invalid(field.NewPath("spec.clusterIPs[0]"), nil, ""),
			},
		},
		// TODO: Add more comprehensive tests.
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

// mkValidService produces a service with a set of tweaks to test validation. 
func mkValidService(tweaks ...func(svc *api.Service)) *api.Service {
    svc := makeValidService()

    for _, tweak := range tweaks {
        tweak(svc)
    }

    return svc
}