/*
Copyright 2023 The Kubernetes Authors.

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

package app

import (
	"context"

	pluginvalidatingadmissionpolicy "k8s.io/apiserver/pkg/admission/plugin/validatingadmissionpolicy"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/controller-manager/controller"
	"k8s.io/kubernetes/pkg/controller/validatingadmissionpolicystatus"
	"k8s.io/kubernetes/pkg/generated/openapi"
)

func startValidatingAdmissionPolicyStatusController(ctx context.Context, controllerContext ControllerContext) (controller.Interface, bool, error) {
	// KCM won't start the controller without the feature gate set.
	typeChecker := &pluginvalidatingadmissionpolicy.TypeChecker{
		SchemaResolver: resolver.NewDefinitionsSchemaResolver(scheme.Scheme, openapi.GetOpenAPIDefinitions),
		RestMapper:     controllerContext.RESTMapper,
	}
	c, err := validatingadmissionpolicystatus.NewController(
		controllerContext.InformerFactory.Admissionregistration().V1beta1().ValidatingAdmissionPolicies(),
		controllerContext.ClientBuilder.ClientOrDie("validatingadmissionpolicy-status-controller").AdmissionregistrationV1beta1().ValidatingAdmissionPolicies(),
		typeChecker,
	)

	go c.Run(ctx, int(controllerContext.ComponentConfig.ValidatingAdmissionPolicyStatusController.ConcurrentPolicySyncs))
	return nil, true, err
}
