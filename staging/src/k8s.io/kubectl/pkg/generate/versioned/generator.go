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

package versioned

import (
	"fmt"
	"io"

	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/kubectl/pkg/generate"
)

// GeneratorFn gives a way to easily override the function for unit testing if needed
var GeneratorFn generate.GeneratorFunc = DefaultGenerators

const (
	// TODO(sig-cli): Enforce consistent naming for generators here.
	// See discussion in https://github.com/kubernetes/kubernetes/issues/46237
	// before you add any more.
	RunPodV1GeneratorName                   = "run-pod/v1"
	ServiceV1GeneratorName                  = "service/v1"
	ServiceV2GeneratorName                  = "service/v2"
	ServiceNodePortGeneratorV1Name          = "service-nodeport/v1"
	ServiceClusterIPGeneratorV1Name         = "service-clusterip/v1"
	ServiceLoadBalancerGeneratorV1Name      = "service-loadbalancer/v1"
	ServiceExternalNameGeneratorV1Name      = "service-externalname/v1"
	ServiceAccountV1GeneratorName           = "serviceaccount/v1"
	HorizontalPodAutoscalerV1GeneratorName  = "horizontalpodautoscaler/v1"
	DeploymentBasicV1Beta1GeneratorName     = "deployment-basic/v1beta1"
	DeploymentBasicAppsV1Beta1GeneratorName = "deployment-basic/apps.v1beta1"
	DeploymentBasicAppsV1GeneratorName      = "deployment-basic/apps.v1"
	NamespaceV1GeneratorName                = "namespace/v1"
	ResourceQuotaV1GeneratorName            = "resourcequotas/v1"
	SecretV1GeneratorName                   = "secret/v1"
	SecretForDockerRegistryV1GeneratorName  = "secret-for-docker-registry/v1"
	SecretForTLSV1GeneratorName             = "secret-for-tls/v1"
	ConfigMapV1GeneratorName                = "configmap/v1"
	ClusterRoleBindingV1GeneratorName       = "clusterrolebinding.rbac.authorization.k8s.io/v1alpha1"
	RoleBindingV1GeneratorName              = "rolebinding.rbac.authorization.k8s.io/v1alpha1"
	PodDisruptionBudgetV1GeneratorName      = "poddisruptionbudget/v1beta1"
	PodDisruptionBudgetV2GeneratorName      = "poddisruptionbudget/v1beta1/v2"
	PriorityClassV1Alpha1GeneratorName      = "priorityclass/v1alpha1"
	PriorityClassV1Beta1GeneratorName       = "priorityclass/v1beta1"
	PriorityClassV1GeneratorName            = "priorityclass/v1"
)

// DefaultGenerators returns the set of default generators for use in Factory instances
func DefaultGenerators(cmdName string) map[string]generate.Generator {
	var generator map[string]generate.Generator
	switch cmdName {
	case "expose":
		generator = map[string]generate.Generator{
			ServiceV1GeneratorName: ServiceGeneratorV1{},
			ServiceV2GeneratorName: ServiceGeneratorV2{},
		}
	case "service-clusterip":
		generator = map[string]generate.Generator{
			ServiceClusterIPGeneratorV1Name: ServiceClusterIPGeneratorV1{},
		}
	case "service-nodeport":
		generator = map[string]generate.Generator{
			ServiceNodePortGeneratorV1Name: ServiceNodePortGeneratorV1{},
		}
	case "service-loadbalancer":
		generator = map[string]generate.Generator{
			ServiceLoadBalancerGeneratorV1Name: ServiceLoadBalancerGeneratorV1{},
		}
	case "deployment":
		// Create Deployment has only StructuredGenerators and no
		// param-based Generators.
		// The StructuredGenerators are as follows (as of 2018-03-16):
		// DeploymentBasicV1Beta1GeneratorName -> DeploymentBasicGeneratorV1
		// DeploymentBasicAppsV1Beta1GeneratorName -> DeploymentBasicAppsGeneratorV1Beta1
		// DeploymentBasicAppsV1GeneratorName -> DeploymentBasicAppsGeneratorV1
		generator = map[string]generate.Generator{}
	case "run":
		generator = map[string]generate.Generator{
			RunPodV1GeneratorName: BasicPod{},
		}
	case "namespace":
		generator = map[string]generate.Generator{
			NamespaceV1GeneratorName: NamespaceGeneratorV1{},
		}
	case "quota":
		generator = map[string]generate.Generator{
			ResourceQuotaV1GeneratorName: ResourceQuotaGeneratorV1{},
		}
	case "secret":
		generator = map[string]generate.Generator{
			SecretV1GeneratorName: SecretGeneratorV1{},
		}
	case "secret-for-docker-registry":
		generator = map[string]generate.Generator{
			SecretForDockerRegistryV1GeneratorName: SecretForDockerRegistryGeneratorV1{},
		}
	case "secret-for-tls":
		generator = map[string]generate.Generator{
			SecretForTLSV1GeneratorName: SecretForTLSGeneratorV1{},
		}
	}

	return generator
}

// FallbackGeneratorNameIfNecessary returns the name of the old generator
// if server does not support new generator. Otherwise, the
// generator string is returned unchanged.
//
// If the generator name is changed, print a warning message to let the user
// know.
func FallbackGeneratorNameIfNecessary(
	generatorName string,
	discoveryClient discovery.DiscoveryInterface,
	cmdErr io.Writer,
) (string, error) {
	switch generatorName {
	case DeploymentBasicAppsV1GeneratorName:
		hasResource, err := HasResource(discoveryClient, appsv1.SchemeGroupVersion.WithResource("deployments"))
		if err != nil {
			return "", err
		}
		if !hasResource {
			return FallbackGeneratorNameIfNecessary(DeploymentBasicAppsV1Beta1GeneratorName, discoveryClient, cmdErr)
		}
	case DeploymentBasicAppsV1Beta1GeneratorName:
		hasResource, err := HasResource(discoveryClient, appsv1beta1.SchemeGroupVersion.WithResource("deployments"))
		if err != nil {
			return "", err
		}
		if !hasResource {
			return DeploymentBasicV1Beta1GeneratorName, nil
		}
	}
	return generatorName, nil
}

func HasResource(client discovery.DiscoveryInterface, resource schema.GroupVersionResource) (bool, error) {
	resources, err := client.ServerResourcesForGroupVersion(resource.GroupVersion().String())
	if apierrors.IsNotFound(err) {
		// entire group is missing
		return false, nil
	}
	if err != nil {
		// other errors error
		return false, fmt.Errorf("failed to discover supported resources: %v", err)
	}
	for _, serverResource := range resources.APIResources {
		if serverResource.Name == resource.Resource {
			return true, nil
		}
	}
	return false, nil
}
