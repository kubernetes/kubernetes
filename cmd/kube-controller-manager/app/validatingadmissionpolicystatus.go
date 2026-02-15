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
	"fmt"

	apiextensionsscheme "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/scheme"
	pluginvalidatingadmissionpolicy "k8s.io/apiserver/pkg/admission/plugin/policy/validating"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	k8sscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-controller-manager/names"
	"k8s.io/kubernetes/pkg/controller/validatingadmissionpolicystatus"
	"k8s.io/kubernetes/pkg/generated/openapi"
)

func newValidatingAdmissionPolicyStatusControllerDescriptor() *ControllerDescriptor {
	return &ControllerDescriptor{
		name:                 names.ValidatingAdmissionPolicyStatusController,
		constructor:          newValidatingAdmissionPolicyStatusController,
		requiredFeatureGates: []featuregate.Feature{},
	}
}

func newValidatingAdmissionPolicyStatusController(ctx context.Context, controllerContext ControllerContext, controllerName string) (Controller, error) {
	discoveryClient, err := controllerContext.ClientBuilder.DiscoveryClient(names.ValidatingAdmissionPolicyStatusController)
	if err != nil {
		return nil, fmt.Errorf("failed to create discovery client for %s: %w", controllerName, err)
	}

	schemaResolver := resolver.NewDefinitionsSchemaResolver(openapi.GetOpenAPIDefinitions, k8sscheme.Scheme, apiextensionsscheme.Scheme).
		Combine(&resolver.ClientDiscoveryResolver{Discovery: discoveryClient})

	typeChecker := &pluginvalidatingadmissionpolicy.TypeChecker{
		SchemaResolver: schemaResolver,
		RestMapper:     controllerContext.RESTMapper,
	}

	client, err := controllerContext.NewClient(names.ValidatingAdmissionPolicyStatusController)
	if err != nil {
		return nil, err
	}

	logger := klog.FromContext(ctx)
	c, err := validatingadmissionpolicystatus.NewController(
		logger,
		controllerContext.InformerFactory.Admissionregistration().V1().ValidatingAdmissionPolicies(),
		client.AdmissionregistrationV1().ValidatingAdmissionPolicies(),
		typeChecker,
	)
	if err != nil {
		return nil, err
	}

	return newControllerLoop(func(ctx context.Context) {
		c.Run(ctx, int(controllerContext.ComponentConfig.ValidatingAdmissionPolicyStatusController.ConcurrentPolicySyncs))
	}, controllerName), nil
}
