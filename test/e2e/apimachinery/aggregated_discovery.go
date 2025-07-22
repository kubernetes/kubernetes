/*
Copyright 2024 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"time"

	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/names"
	clientdiscovery "k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("AggregatedDiscovery", func() {
	f := framework.NewDefaultFramework("aggregateddiscovery")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// Ensure that resources in both the legacy api/v1 group-version and resources in /apis/* exist
	expectedLegacyGVR := []schema.GroupVersionResource{
		{
			Group:    "",
			Version:  "v1",
			Resource: "namespaces",
		},
	}
	expectedGVR := []schema.GroupVersionResource{
		{
			Group:    "admissionregistration.k8s.io",
			Version:  "v1",
			Resource: "validatingwebhookconfigurations",
		},
		{
			Group:    "apiextensions.k8s.io",
			Version:  "v1",
			Resource: "customresourcedefinitions",
		},
		{
			Group:    "apiregistration.k8s.io",
			Version:  "v1",
			Resource: "apiservices",
		},
		{
			Group:    "apps",
			Version:  "v1",
			Resource: "deployments",
		},
		{
			Group:    "authentication.k8s.io",
			Version:  "v1",
			Resource: "tokenreviews",
		},
		{
			Group:    "authorization.k8s.io",
			Version:  "v1",
			Resource: "selfsubjectaccessreviews",
		},
		{
			Group:    "autoscaling",
			Version:  "v1",
			Resource: "horizontalpodautoscalers",
		},
		{
			Group:    "autoscaling",
			Version:  "v2",
			Resource: "horizontalpodautoscalers",
		},
		{
			Group:    "batch",
			Version:  "v1",
			Resource: "jobs",
		},
		{
			Group:    "certificates.k8s.io",
			Version:  "v1",
			Resource: "certificatesigningrequests",
		},
		{
			Group:    "coordination.k8s.io",
			Version:  "v1",
			Resource: "leases",
		},
		{
			Group:    "discovery.k8s.io",
			Version:  "v1",
			Resource: "endpointslices",
		},
		{
			Group:    "events.k8s.io",
			Version:  "v1",
			Resource: "events",
		},
		{
			Group:    "networking.k8s.io",
			Version:  "v1",
			Resource: "ingresses",
		},
		{
			Group:    "node.k8s.io",
			Version:  "v1",
			Resource: "runtimeclasses",
		},
		{
			Group:    "policy",
			Version:  "v1",
			Resource: "poddisruptionbudgets",
		},
		{
			Group:    "scheduling.k8s.io",
			Version:  "v1",
			Resource: "priorityclasses",
		},
		{
			Group:    "storage.k8s.io",
			Version:  "v1",
			Resource: "csinodes",
		},
	}

	const aggregatedAccept = "application/json;g=apidiscovery.k8s.io;v=v2;as=APIGroupDiscoveryList"

	/*
		Release : v1.30
		Testname: Aggregated Discovery Endpoint Accept Headers
		Description: An apiserver MUST support the Aggregated Discovery endpoint Accept headers. Built-in resources MUST all be present.
	*/
	framework.ConformanceIt("should support raw aggregated discovery endpoint Accept headers", func(ctx context.Context) {
		d, err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis").SetHeader("Accept", aggregatedAccept).Do(ctx).Raw()
		if err != nil {
			framework.Failf("Failed to get raw aggregated discovery document")
		}

		groupList := apidiscoveryv2.APIGroupDiscoveryList{}
		err = json.Unmarshal(d, &groupList)
		if err != nil {
			framework.Failf("Failed to parse discovery: %v", err)
		}

		for _, gvr := range expectedGVR {
			if !isGVRPresentAPIDiscovery(groupList, gvr) {
				framework.Failf("Expected gvr %s %s %s to exist in discovery", gvr.Group, gvr.Version, gvr.Resource)

			}
		}

		d2, err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/api").SetHeader("Accept", aggregatedAccept).Do(ctx).Raw()
		if err != nil {
			framework.Failf("Failed to get raw aggregated discovery document")
		}

		groupListLegacy := apidiscoveryv2.APIGroupDiscoveryList{}
		err = json.Unmarshal(d2, &groupListLegacy)
		if err != nil {
			framework.Failf("Failed to parse discovery: %v", err)
		}

		for _, gvr := range expectedLegacyGVR {
			if !isGVRPresentAPIDiscovery(groupListLegacy, gvr) {
				framework.Failf("Expected legacy gvr api %s %s to exist in discovery", gvr.Version, gvr.Resource)
			}
		}
	})

	/*
		Release : v1.30
		Testname: Aggregated Discovery Endpoint Accept Headers CRDs
		Description: An apiserver MUST support the Aggregated Discovery endpoint Accept headers.
		Add a CRD to the apiserver. The CRD MUST appear in the discovery document.
	*/
	framework.ConformanceIt("should support raw aggregated discovery request for CRDs", func(ctx context.Context) {
		config, err := framework.LoadConfig()
		framework.ExpectNoError(err)
		apiExtensionClient, err := apiextensionclientset.NewForConfig(config)
		framework.ExpectNoError(err)
		dynamicClient, err := dynamic.NewForConfig(config)
		framework.ExpectNoError(err)
		resourceName := "testcrd"
		// Generate a CRD with random group name to avoid group conflict with other tests that run in parallel.
		groupName := fmt.Sprintf("%s.example.com", names.SimpleNameGenerator.GenerateName("group"))
		crd := &apiextensionsv1.CustomResourceDefinition{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("%ss.%s", resourceName, groupName)},
			Spec: apiextensionsv1.CustomResourceDefinitionSpec{
				Group: groupName,
				Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
					{
						Name:    "v1",
						Served:  true,
						Storage: true,
						Schema:  fixtures.AllowAllSchema(),
					},
				},
				Names: apiextensionsv1.CustomResourceDefinitionNames{
					Plural:   resourceName + "s",
					Singular: resourceName,
					Kind:     resourceName,
					ListKind: resourceName + "List",
				},
				Scope: apiextensionsv1.NamespaceScoped,
			},
		}
		gvr := schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: resourceName + "s"}
		_, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
		framework.ExpectNoError(err)
		defer func() {
			_ = fixtures.DeleteV1CustomResourceDefinition(crd, apiExtensionClient)
		}()

		err = wait.PollUntilContextTimeout(context.Background(), time.Second*1, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {

			d, err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis").SetHeader("Accept", aggregatedAccept).Do(ctx).Raw()
			if err != nil {
				framework.Failf("Failed to get raw aggregated discovery document")
			}

			groupList := apidiscoveryv2.APIGroupDiscoveryList{}
			err = json.Unmarshal(d, &groupList)
			if err != nil {
				framework.Failf("Failed to parse discovery: %v", err)
			}
			if isGVRPresentAPIDiscovery(groupList, gvr) {
				return true, nil
			}
			return false, nil

		})
		framework.ExpectNoError(err, "timed out waiting for CustomResourceDefinition GVR to appear in Discovery")

	})

	/*
		Release : v1.30
		Testname: Aggregated Discovery Interface
		Description: An apiserver MUST support the Aggregated Discovery client interface. Built-in resources MUST all be present.
	*/
	framework.ConformanceIt("should support aggregated discovery interface", func(ctx context.Context) {
		d := f.ClientSet.Discovery()

		ad, ok := d.(clientdiscovery.AggregatedDiscoveryInterface)
		if !ok {
			framework.Failf("Expected client to support aggregated discovery")
		}
		serverGroups, resourcesByGV, _, err := ad.GroupsAndMaybeResources()
		if err != nil {
			framework.Failf("Failed to get api groups and resources: %v", err)
		}

		expectedCombinedGVR := append(expectedGVR, expectedLegacyGVR...)
		expectedGVs := []schema.GroupVersion{}
		for _, gvr := range expectedCombinedGVR {
			expectedGVs = append(expectedGVs, schema.GroupVersion{
				Group:   gvr.Group,
				Version: gvr.Version,
			})
		}

		for _, gvr := range expectedCombinedGVR {
			if !isGVRPresent(resourcesByGV, gvr) {
				framework.Failf("Expected %v to be present", gvr)
			}
		}

		if serverGroups == nil {
			framework.Failf("Expected serverGroups to be non-nil")
		}

		for _, gv := range expectedGVs {
			if !isGVPresent(serverGroups, gv) {
				framework.Failf("Expected %v to be present", gv)
			}
		}
	})

	/*
		Release : v1.30
		Testname: Aggregated Discovery Interface CRDs
		Description: An apiserver MUST support the Aggregated Discovery client interface.
		Add a CRD to the apiserver. The CRD resource MUST be present in the discovery document.
	*/
	framework.ConformanceIt("should support aggregated discovery interface for CRDs", func(ctx context.Context) {
		config, err := framework.LoadConfig()
		framework.ExpectNoError(err)
		apiExtensionClient, err := apiextensionclientset.NewForConfig(config)
		framework.ExpectNoError(err)
		dynamicClient, err := dynamic.NewForConfig(config)
		framework.ExpectNoError(err)
		resourceName := "testcrd"
		// Generate a CRD with random group name to avoid group conflict with other tests that run in parallel.
		groupName := fmt.Sprintf("%s.example.com", names.SimpleNameGenerator.GenerateName("group"))
		crd := &apiextensionsv1.CustomResourceDefinition{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("%ss.%s", resourceName, groupName)},
			Spec: apiextensionsv1.CustomResourceDefinitionSpec{
				Group: groupName,
				Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
					{
						Name:    "v1",
						Served:  true,
						Storage: true,
						Schema:  fixtures.AllowAllSchema(),
					},
				},
				Names: apiextensionsv1.CustomResourceDefinitionNames{
					Plural:   resourceName + "s",
					Singular: resourceName,
					Kind:     resourceName,
					ListKind: resourceName + "List",
				},
				Scope: apiextensionsv1.NamespaceScoped,
			},
		}
		gvr := schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: resourceName + "s"}
		_, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
		framework.ExpectNoError(err)
		defer func() {
			_ = fixtures.DeleteV1CustomResourceDefinition(crd, apiExtensionClient)
		}()

		d := f.ClientSet.Discovery()

		ad, ok := d.(clientdiscovery.AggregatedDiscoveryInterface)
		if !ok {
			framework.Failf("Expected client to support aggregated discovery")
		}
		err = wait.PollUntilContextTimeout(context.Background(), time.Second*1, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
			_, resourcesByGV, _, err := ad.GroupsAndMaybeResources()
			if err != nil {
				framework.Failf("Failed to get api groups and resources: %v", err)
			}
			if isGVRPresent(resourcesByGV, gvr) {
				return true, nil
			}
			return false, nil

		})
		framework.ExpectNoError(err, "timed out waiting for CustomResourceDefinition GVR to appear in Discovery")
	})
})

func isGVRPresent(resourcesByGV map[schema.GroupVersion]*metav1.APIResourceList, gvr schema.GroupVersionResource) bool {
	resourceList, ok := resourcesByGV[gvr.GroupVersion()]
	if !ok {
		return false
	}

	if resourceList == nil {
		return false
	}
	for _, resource := range resourceList.APIResources {
		if resource.Name == gvr.Resource {
			return true
		}
	}
	return false
}

func isGVPresent(gvs *metav1.APIGroupList, gv schema.GroupVersion) bool {
	for _, group := range gvs.Groups {
		if group.Name != gv.Group {
			continue
		}
		for _, version := range group.Versions {
			if version.Version == gv.Version {
				return true
			}
		}
	}
	return false
}

func isGVRPresentAPIDiscovery(apidiscovery apidiscoveryv2.APIGroupDiscoveryList, gvr schema.GroupVersionResource) bool {
	for _, group := range apidiscovery.Items {
		if gvr.Group == group.Name {
			for _, version := range group.Versions {
				if version.Version == gvr.Version {
					for _, resource := range version.Resources {
						if resource.Resource == gvr.Resource {
							return true
						}
					}
				}
			}
		}
	}
	return false
}
