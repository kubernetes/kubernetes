/*
Copyright 2021 The Kubernetes Authors.

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
	"reflect"

	appsv1 "k8s.io/api/apps/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"

	// ensure libs have a chance to initialize
	_ "github.com/stretchr/testify/assert"
)

var _ = SIGDescribe("ServerSideApply", func() {
	f := framework.NewDefaultFramework("apply")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var client clientset.Interface
	var ns string

	ginkgo.BeforeEach(func() {
		client = f.ClientSet
		ns = f.Namespace.Name
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		_ = client.AppsV1().Deployments(ns).Delete(ctx, "deployment", metav1.DeleteOptions{})
		_ = client.AppsV1().Deployments(ns).Delete(ctx, "deployment-shared-unset", metav1.DeleteOptions{})
		_ = client.AppsV1().Deployments(ns).Delete(ctx, "deployment-shared-map-item-removal", metav1.DeleteOptions{})
		_ = client.CoreV1().Pods(ns).Delete(ctx, "test-pod", metav1.DeleteOptions{})
	})

	/*
		Release : v1.21
		Testname: Server Side Apply, Create
		Description: Apply an object. An apply on an object that does not exist MUST create the object.
	*/
	ginkgo.It("should create an applied object if it does not already exist", func(ctx context.Context) {
		testCases := []struct {
			resource      string
			name          string
			body          string
			managedFields string
		}{
			{
				resource: "pods",
				name:     "test-pod",
				body: `{
				"apiVersion": "v1",
				"kind": "Pod",
				"metadata": {
					"name": "test-pod"
				},
				"spec": {
					"containers": [{
						"name":  "test-container",
						"image": "test-image"
					}]
				}
			}`,
				managedFields: `{"f:spec":{"f:containers":{"k:{\"name\":\"test-container\"}":{".":{},"f:image":{},"f:name":{}}}}}`,
			}, {
				resource: "services",
				name:     "test-svc",
				body: `{
				"apiVersion": "v1",
				"kind": "Service",
				"metadata": {
					"name": "test-svc"
				},
				"spec": {
					"ports": [{
						"port": 8080,
						"protocol": "UDP"
					}]
				}
			}`,
				managedFields: `{"f:spec":{"f:ports":{"k:{\"port\":8080,\"protocol\":\"UDP\"}":{".":{},"f:port":{},"f:protocol":{}}}}}`,
			},
		}

		for _, tc := range testCases {
			_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
				Namespace(ns).
				Resource(tc.resource).
				Name(tc.name).
				Param("fieldManager", "apply_test").
				Body([]byte(tc.body)).
				Do(ctx).
				Get()
			if err != nil {
				framework.Failf("Failed to create object using Apply patch: %v", err)
			}

			_, err = client.CoreV1().RESTClient().Get().Namespace(ns).Resource(tc.resource).Name(tc.name).Do(ctx).Get()
			if err != nil {
				framework.Failf("Failed to retrieve object: %v", err)
			}

			// Test that we can re apply with a different field manager and don't get conflicts
			obj, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
				Namespace(ns).
				Resource(tc.resource).
				Name(tc.name).
				Param("fieldManager", "apply_test_2").
				Body([]byte(tc.body)).
				Do(ctx).
				Get()
			if err != nil {
				framework.Failf("Failed to re-apply object using Apply patch: %v", err)
			}

			// Verify that both appliers own the fields
			accessor, err := meta.Accessor(obj)
			framework.ExpectNoError(err, "getting ObjectMeta")
			managedFields := accessor.GetManagedFields()
			for _, entry := range managedFields {
				if entry.Manager == "apply_test_2" || entry.Manager == "apply_test" {
					if entry.FieldsV1.String() != tc.managedFields {
						framework.Failf("Expected managed fields %s, got %s", tc.managedFields, entry.FieldsV1.String())
					}
				}
			}
		}
	})

	/*
		Release : v1.21
		Testname: Server Side Apply, Subresource
		Description: Apply a resource and issue a subsequent apply on a subresource. The subresource MUST be updated with the applied object contents.
	*/
	ginkgo.It("should work for subresources", func(ctx context.Context) {
		{
			testCases := []struct {
				resource    string
				name        string
				body        string
				statusPatch string
			}{
				{
					resource: "pods",
					name:     "test-pod",
					body: `{
				"apiVersion": "v1",
				"kind": "Pod",
				"metadata": {
					"name": "test-pod"
				},
				"spec": {
					"containers": [{
						"name":  "nginx",
						"image": "` + imageutils.GetE2EImage(imageutils.NginxNew) + `",
					}]
				}
			}`,
					statusPatch: `{
				"apiVersion": "v1",
				"kind": "Pod",
				"metadata": {
					"name": "test-pod"
				},
				"status": {"conditions": [{"type": "MyStatus", "status":"True"}]}}`,
				},
			}

			for _, tc := range testCases {
				_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
					Namespace(ns).
					Resource(tc.resource).
					Name(tc.name).
					Param("fieldManager", "apply_test").
					Body([]byte(tc.body)).
					Do(ctx).
					Get()
				if err != nil {
					framework.Failf("Failed to create object using Apply patch: %v", err)
				}

				_, err = client.CoreV1().RESTClient().Get().Namespace(ns).Resource(tc.resource).Name(tc.name).Do(ctx).Get()
				if err != nil {
					framework.Failf("Failed to retrieve object: %v", err)
				}

				// Test that apply does not update subresources unless directed at a subresource endpoint
				_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
					Namespace(ns).
					Resource(tc.resource).
					Name(tc.name).
					Param("fieldManager", "apply_test2").
					Body([]byte(tc.statusPatch)).
					Do(ctx).
					Get()
				if err != nil {
					framework.Failf("Failed to Apply Status using Apply patch: %v", err)
				}
				pod, err := client.CoreV1().Pods(ns).Get(ctx, "test-pod", metav1.GetOptions{})
				framework.ExpectNoError(err, "retrieving test pod")
				for _, c := range pod.Status.Conditions {
					if c.Type == "MyStatus" {
						framework.Failf("Apply should not update subresources unless the endpoint is specifically specified")
					}
				}

				// Test that apply to subresource updates the subresource
				_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
					Namespace(ns).
					Resource(tc.resource).
					SubResource("status").
					Name(tc.name).
					Param("fieldManager", "apply_test2").
					Body([]byte(tc.statusPatch)).
					Do(ctx).
					Get()
				if err != nil {
					framework.Failf("Failed to Apply Status using Apply patch: %v", err)
				}

				pod, err = client.CoreV1().Pods(ns).Get(ctx, "test-pod", metav1.GetOptions{})
				framework.ExpectNoError(err, "retrieving test pod")

				myStatusFound := false
				for _, c := range pod.Status.Conditions {
					if c.Type == "MyStatus" {
						myStatusFound = true
						break
					}
				}
				if myStatusFound == false {
					framework.Failf("Expected pod to have applied status")
				}
			}
		}
	})

	/*
		Release : v1.21
		Testname: Server Side Apply, unset field
		Description: Apply an object. Issue a subsequent apply that removes a field. The particular field MUST be removed.
	*/
	ginkgo.It("should remove a field if it is owned but removed in the apply request", func(ctx context.Context) {
		obj := []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"replicas": 3,
			"selector": {
				"matchLabels": {
					"app": "nginx"
				}
			},
			"template": {
				"metadata": {
					"labels": {
						"app": "nginx"
					}
				},
				"spec": {
					"containers": [{
						"name":  "nginx",
						"image": "` + imageutils.GetE2EImage(imageutils.NginxNew) + `",
						"ports": [{
							"containerPort": 80,
							"protocol": "TCP"
						}]
					}]
				}
			}
		}
	}`)

		_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
			AbsPath("/apis/apps/v1").
			Namespace(ns).
			Resource("deployments").
			Name("deployment").
			Param("fieldManager", "apply_test").
			Body(obj).Do(ctx).Get()
		if err != nil {
			framework.Failf("Failed to create object using Apply patch: %v", err)
		}

		obj = []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"replicas": 3,
			"selector": {
				"matchLabels": {
					"app": "nginx"
				}
			},
			"template": {
				"metadata": {
					"labels": {
						"app": "nginx"
					}
				},
				"spec": {
					"containers": [{
						"name":  "nginx",
						"image": "` + imageutils.GetE2EImage(imageutils.NginxNew) + `",
					}]
				}
			}
		}
	}`)

		_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
			AbsPath("/apis/apps/v1").
			Namespace(ns).
			Resource("deployments").
			Name("deployment").
			Param("fieldManager", "apply_test").
			Body(obj).Do(ctx).Get()
		if err != nil {
			framework.Failf("Failed to remove container port using Apply patch: %v", err)
		}

		deployment, err := client.AppsV1().Deployments(ns).Get(ctx, "deployment", metav1.GetOptions{})
		if err != nil {
			framework.Failf("Failed to retrieve object: %v", err)
		}

		if len(deployment.Spec.Template.Spec.Containers[0].Ports) > 0 {
			framework.Failf("Expected no container ports but got: %v, object: \n%#v", deployment.Spec.Template.Spec.Containers[0].Ports, deployment)
		}

	})

	/*
		Release : v1.21
		Testname: Server Side Apply, unset field shared
		Description: Apply an object. Unset ownership of a field that is also owned by other managers and make a subsequent apply request. The unset field MUST not be removed from the object.
	*/
	ginkgo.It("should not remove a field if an owner unsets the field but other managers still have ownership of the field", func(ctx context.Context) {
		// spec.replicas is a optional, defaulted field
		// spec.template.spec.hostname is an optional, non-defaulted field
		apply := []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment-shared-unset",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"replicas": 3,
			"selector": {
				"matchLabels": {
					"app": "nginx"
				}
			},
			"template": {
				"metadata": {
					"labels": {
						"app": "nginx"
					}
				},
				"spec": {
					"hostname": "test-hostname",
					"containers": [{
						"name":  "nginx",
						"image": "` + imageutils.GetE2EImage(imageutils.NginxNew) + `",
					}]
				}
			}
		}
	}`)

		for _, fieldManager := range []string{"shared_owner_1", "shared_owner_2"} {
			_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
				AbsPath("/apis/apps/v1").
				Namespace(ns).
				Resource("deployments").
				Name("deployment-shared-unset").
				Param("fieldManager", fieldManager).
				Body(apply).
				Do(ctx).
				Get()
			if err != nil {
				framework.Failf("Failed to create object using Apply patch: %v", err)
			}
		}

		// unset spec.replicas and spec.template.spec.hostname
		apply = []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment-shared-unset",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"selector": {
				"matchLabels": {
					"app": "nginx"
				}
			},
			"template": {
				"metadata": {
					"labels": {
						"app": "nginx"
					}
				},
				"spec": {
					"containers": [{
						"name":  "nginx",
						"image": "` + imageutils.GetE2EImage(imageutils.NginxNew) + `",
					}]
				}
			}
		}
	}`)

		patched, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
			AbsPath("/apis/apps/v1").
			Namespace(ns).
			Resource("deployments").
			Name("deployment-shared-unset").
			Param("fieldManager", "shared_owner_1").
			Body(apply).
			Do(ctx).
			Get()
		if err != nil {
			framework.Failf("Failed to create object using Apply patch: %v", err)
		}

		deployment, ok := patched.(*appsv1.Deployment)
		if !ok {
			framework.Failf("Failed to convert response object to Deployment")
		}
		if *deployment.Spec.Replicas != 3 {
			framework.Failf("Expected deployment.spec.replicas to be 3, but got %d", deployment.Spec.Replicas)
		}
		if deployment.Spec.Template.Spec.Hostname != "test-hostname" {
			framework.Failf("Expected deployment.spec.template.spec.hostname to be \"test-hostname\", but got %s", deployment.Spec.Template.Spec.Hostname)
		}
	})

	/*
		Release : v1.21
		Testname: Server Side Apply, Force Apply
		Description: Apply an object. Force apply a modified version of the object such that a conflict will exist in the managed fields. The force apply MUST successfully update the object.
	*/
	ginkgo.It("should ignore conflict errors if force apply is used", func(ctx context.Context) {
		obj := []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"replicas": 3,
			"selector": {
				"matchLabels": {
					 "app": "nginx"
				}
			},
			"template": {
				"metadata": {
					"labels": {
						"app": "nginx"
					}
				},
				"spec": {
					"containers": [{
						"name":  "nginx",
						"image": "` + imageutils.GetE2EImage(imageutils.NginxNew) + `",
					}]
				}
			}
		}
	}`)
		_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
			AbsPath("/apis/apps/v1").
			Namespace(ns).
			Resource("deployments").
			Name("deployment").
			Param("fieldManager", "apply_test").
			Body(obj).Do(ctx).Get()
		if err != nil {
			framework.Failf("Failed to create object using Apply patch: %v", err)
		}

		_, err = client.CoreV1().RESTClient().Patch(types.MergePatchType).
			AbsPath("/apis/apps/v1").
			Namespace(ns).
			Resource("deployments").
			Name("deployment").
			Body([]byte(`{"spec":{"replicas": 5}}`)).Do(ctx).Get()
		if err != nil {
			framework.Failf("Failed to patch object: %v", err)
		}

		_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
			AbsPath("/apis/apps/v1").
			Namespace(ns).
			Resource("deployments").
			Name("deployment").
			Param("fieldManager", "apply_test").
			Body(obj).Do(ctx).Get()
		if err == nil {
			framework.Failf("Expecting to get conflicts when applying object")
		}
		status, ok := err.(*apierrors.StatusError)
		if !(ok && apierrors.IsConflict(status)) {
			framework.Failf("Expecting to get conflicts as API error")
		}
		if len(status.Status().Details.Causes) < 1 {
			framework.Failf("Expecting to get at least one conflict when applying object, got: %v", status.Status().Details.Causes)
		}

		_, err = client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
			AbsPath("/apis/apps/v1").
			Namespace(ns).
			Resource("deployments").
			Name("deployment").
			Param("force", "true").
			Param("fieldManager", "apply_test").
			Body(obj).Do(ctx).Get()
		if err != nil {
			framework.Failf("Failed to apply object with force: %v", err)
		}
	})

	/*
		Release : v1.21
		Testname: Server Side Apply, CRD
		Description: Create a CRD and apply a CRD resource. Subsequent apply requests that do not conflict with the previous ones should update the object. Apply requests that cause conflicts should fail.
	*/
	ginkgo.It("should work for CRDs", func(ctx context.Context) {
		config, err := framework.LoadConfig()
		if err != nil {
			framework.Failf("%s", err)
		}
		apiExtensionClient, err := apiextensionclientset.NewForConfig(config)
		if err != nil {
			framework.Failf("%s", err)
		}
		dynamicClient, err := dynamic.NewForConfig(config)
		if err != nil {
			framework.Failf("%s", err)
		}

		noxuDefinition := fixtures.NewRandomNameMultipleVersionCustomResourceDefinition(apiextensionsv1.ClusterScoped)

		var c apiextensionsv1.CustomResourceValidation
		err = json.Unmarshal([]byte(`{
		"openAPIV3Schema": {
			"type": "object",
			"properties": {
				"spec": {
					"type": "object",
					"x-kubernetes-preserve-unknown-fields": true,
					"properties": {
						"cronSpec": {
							"type": "string",
							"pattern": "^(\\d+|\\*)(/\\d+)?(\\s+(\\d+|\\*)(/\\d+)?){4}$"
						},
						"ports": {
							"type": "array",
							"x-kubernetes-list-map-keys": [
								"containerPort",
								"protocol"
							],
							"x-kubernetes-list-type": "map",
							"items": {
								"properties": {
									"containerPort": {
										"format": "int32",
										"type": "integer"
									},
									"hostIP": {
										"type": "string"
									},
									"hostPort": {
										"format": "int32",
										"type": "integer"
									},
									"name": {
										"type": "string"
									},
									"protocol": {
										"type": "string"
									}
								},
								"required": [
									"containerPort",
									"protocol"
								],
								"type": "object"
							}
						}
					}
				}
			}
		}
	}`), &c)
		if err != nil {
			framework.Failf("%s", err)
		}
		for i := range noxuDefinition.Spec.Versions {
			noxuDefinition.Spec.Versions[i].Schema = &c
		}

		noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
		if err != nil {
			framework.Failf("cannot create crd %s", err)
		}

		defer func() {
			err = fixtures.DeleteV1CustomResourceDefinition(noxuDefinition, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinition")
		}()

		kind := noxuDefinition.Spec.Names.Kind
		apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Versions[0].Name
		name := "mytest"

		rest := apiExtensionClient.Discovery().RESTClient()
		yamlBody := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
  finalizers:
  - test-finalizer
spec:
  cronSpec: "* * * * */5"
  replicas: 1
  ports:
  - name: x
    containerPort: 80
    protocol: TCP`, apiVersion, kind, name))
		result, err := rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "apply_test").
			Body(yamlBody).
			DoRaw(ctx)
		if err != nil {
			framework.Failf("failed to create custom resource with apply: %v:\n%v", err, string(result))
		}
		verifyNumFinalizers(result, 1)
		verifyFinalizersIncludes(result, "test-finalizer")
		verifyReplicas(result, 1)
		verifyNumPorts(result, 1)

		// Ensure that apply works with multiple resource versions
		apiVersionBeta := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Versions[1].Name
		yamlBodyBeta := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
spec:
  cronSpec: "* * * * */5"
  replicas: 1
  ports:
  - name: x
    containerPort: 80
    protocol: TCP`, apiVersionBeta, kind, name))
		result, err = rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[1].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "apply_test").
			Body(yamlBodyBeta).
			DoRaw(ctx)
		if err != nil {
			framework.Failf("failed to create custom resource with apply: %v:\n%v", err, string(result))
		}
		verifyReplicas(result, 1)
		verifyNumPorts(result, 1)

		// Reset the finalizers after the test so the objects can be deleted
		defer func() {
			result, err = rest.Patch(types.MergePatchType).
				AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
				Name(name).
				Body([]byte(`{"metadata":{"finalizers":[]}}`)).
				DoRaw(ctx)
			if err != nil {
				framework.Failf("failed to reset finalizers: %v:\n%v", err, string(result))
			}
		}()

		// Patch object to add another finalizer to the finalizers list
		result, err = rest.Patch(types.MergePatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Body([]byte(`{"metadata":{"finalizers":["test-finalizer","another-one"]}}`)).
			DoRaw(ctx)
		if err != nil {
			framework.Failf("failed to add finalizer with merge patch: %v:\n%v", err, string(result))
		}
		verifyNumFinalizers(result, 2)
		verifyFinalizersIncludes(result, "test-finalizer")
		verifyFinalizersIncludes(result, "another-one")

		// Re-apply the same config, should work fine, since finalizers should have the list-type extension 'set'.
		result, err = rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "apply_test").
			SetHeader("Accept", "application/json").
			Body(yamlBody).
			DoRaw(ctx)
		if err != nil {
			framework.Failf("failed to apply same config after adding a finalizer: %v:\n%v", err, string(result))
		}
		verifyNumFinalizers(result, 2)
		verifyFinalizersIncludes(result, "test-finalizer")
		verifyFinalizersIncludes(result, "another-one")

		// Patch object to change the number of replicas
		result, err = rest.Patch(types.MergePatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Body([]byte(`{"spec":{"replicas": 5}}`)).
			DoRaw(ctx)
		if err != nil {
			framework.Failf("failed to update number of replicas with merge patch: %v:\n%v", err, string(result))
		}
		verifyReplicas(result, 5)

		// Re-apply, we should get conflicts now, since the number of replicas was changed.
		result, err = rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "apply_test").
			Body(yamlBody).
			DoRaw(ctx)
		if err == nil {
			framework.Failf("Expecting to get conflicts when applying object after updating replicas, got no error: %s", result)
		}
		status, ok := err.(*apierrors.StatusError)
		if !ok {
			framework.Failf("Expecting to get conflicts as API error")
		}
		if len(status.Status().Details.Causes) != 1 {
			framework.Failf("Expecting to get one conflict when applying object after updating replicas, got: %v", status.Status().Details.Causes)
		}

		// Re-apply with force, should work fine.
		result, err = rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Param("force", "true").
			Param("fieldManager", "apply_test").
			Body(yamlBody).
			DoRaw(ctx)
		if err != nil {
			framework.Failf("failed to apply object with force after updating replicas: %v:\n%v", err, string(result))
		}
		verifyReplicas(result, 1)

		// New applier tries to edit an existing list item, we should get conflicts.
		result, err = rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "apply_test_2").
			Body([]byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
spec:
  ports:
  - name: "y"
    containerPort: 80
    protocol: TCP`, apiVersion, kind, name))).
			DoRaw(ctx)
		if err == nil {
			framework.Failf("Expecting to get conflicts when a different applier updates existing list item, got no error: %s", result)
		}
		status, ok = err.(*apierrors.StatusError)
		if !ok {
			framework.Failf("Expecting to get conflicts as API error")
		}
		if len(status.Status().Details.Causes) != 1 {
			framework.Failf("Expecting to get one conflict when a different applier updates existing list item, got: %v", status.Status().Details.Causes)
		}

		// New applier tries to add a new list item, should work fine.
		result, err = rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "apply_test_2").
			Body([]byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
spec:
  ports:
  - name: "y"
    containerPort: 8080
    protocol: TCP`, apiVersion, kind, name))).
			SetHeader("Accept", "application/json").
			DoRaw(ctx)
		if err != nil {
			framework.Failf("failed to add a new list item to the object as a different applier: %v:\n%v", err, string(result))
		}
		verifyNumPorts(result, 2)

		// UpdateOnCreate
		notExistingYAMLBody := []byte(fmt.Sprintf(`
	{
		"apiVersion": "%s",
		"kind": "%s",
		"metadata": {
		  "name": "%s",
		  "finalizers": [
			"test-finalizer"
		  ]
		},
		"spec": {
		  "cronSpec": "* * * * */5",
		  "replicas": 1,
		  "ports": [
			{
			  "name": "x",
			  "containerPort": 80
			}
		  ]
		},
		"protocol": "TCP"
	}`, apiVersion, kind, "should-not-exist"))
		_, err = rest.Put().
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Name("should-not-exist").
			Param("fieldManager", "apply_test").
			Body(notExistingYAMLBody).
			DoRaw(ctx)
		if !apierrors.IsNotFound(err) {
			framework.Failf("create on update should fail with notFound, got %v", err)
		}

		// Create a CRD to test atomic lists
		crd := fixtures.NewRandomNameV1CustomResourceDefinition(apiextensionsv1.ClusterScoped)
		err = json.Unmarshal([]byte(`{
		"openAPIV3Schema": {
			"type": "object",
			"properties": {
				"spec": {
					"type": "object",
					"x-kubernetes-preserve-unknown-fields": true,
					"properties": {
						"atomicList": {
							"type": "array",
							"x-kubernetes-list-type": "atomic",
							"items": {
								"type": "string"
							}
						}
					}
				}
			}
		}
	}`), &c)
		if err != nil {
			framework.Failf("%s", err)
		}
		for i := range crd.Spec.Versions {
			crd.Spec.Versions[i].Schema = &c
		}

		crd, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
		if err != nil {
			framework.Failf("cannot create crd %s", err)
		}

		defer func() {
			err = fixtures.DeleteV1CustomResourceDefinition(crd, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinition")
		}()

		crdKind := crd.Spec.Names.Kind
		crdApiVersion := crd.Spec.Group + "/" + crd.Spec.Versions[0].Name

		crdYamlBody := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
spec:
  atomicList:
  - "item1"`, crdApiVersion, crdKind, name))
		result, err = rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", crd.Spec.Group, crd.Spec.Versions[0].Name, crd.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "apply_test").
			Body(crdYamlBody).
			DoRaw(ctx)
		if err != nil {
			framework.Failf("failed to create custom resource with apply: %v:\n%v", err, string(result))
		}

		verifyList(result, []interface{}{"item1"})

		crdYamlBody = []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: %s
spec:
  atomicList:
  - "item2"`, crdApiVersion, crdKind, name))
		result, err = rest.Patch(types.ApplyPatchType).
			AbsPath("/apis", crd.Spec.Group, crd.Spec.Versions[0].Name, crd.Spec.Names.Plural).
			Name(name).
			Param("fieldManager", "apply_test_2").
			Param("force", "true").
			Body(crdYamlBody).
			DoRaw(ctx)
		if err != nil {
			framework.Failf("failed to create custom resource with apply: %v:\n%v", err, string(result))
		}

		// Since the list is atomic the contents of the list must completely be replaced by the latest apply
		verifyList(result, []interface{}{"item2"})
	})

	/*
		Release : v1.21
		Testname: Server Side Apply, Update take ownership
		Description: Apply an object. Send an Update request which should take ownership of a field. The field should be owned by the new manager and a subsequent apply from the original manager MUST not change the field it does not have ownership of.
	*/
	ginkgo.It("should give up ownership of a field if forced applied by a controller", func(ctx context.Context) {
		// Applier creates a deployment with replicas set to 3
		apply := []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment-shared-map-item-removal",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"replicas": 3,
			"selector": {
				"matchLabels": {
					"app": "nginx"
				}
			},
			"template": {
				"metadata": {
					"labels": {
						"app": "nginx"
					}
				},
				"spec": {
					"containers": [{
						"name":  "nginx",
						"image": "` + imageutils.GetE2EImage(imageutils.NginxNew) + `",
					}]
				}
			}
		}
	}`)

		_, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
			AbsPath("/apis/apps/v1").
			Namespace(ns).
			Resource("deployments").
			Name("deployment-shared-map-item-removal").
			Param("fieldManager", "test_applier").
			Body(apply).
			Do(ctx).
			Get()
		if err != nil {
			framework.Failf("Failed to create object using Apply patch: %v", err)
		}

		replicas := int32(4)
		_, err = e2edeployment.UpdateDeploymentWithRetries(client, ns, "deployment-shared-map-item-removal", func(update *appsv1.Deployment) {
			update.Spec.Replicas = &replicas
		})
		framework.ExpectNoError(err)

		// applier omits replicas
		apply = []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment-shared-map-item-removal",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"selector": {
				"matchLabels": {
					"app": "nginx"
				}
			},
			"template": {
				"metadata": {
					"labels": {
						"app": "nginx"
					}
				},
				"spec": {
					"containers": [{
						"name":  "nginx",
						"image": "` + imageutils.GetE2EImage(imageutils.NginxNew) + `",
					}]
				}
			}
		}
	}`)

		patched, err := client.CoreV1().RESTClient().Patch(types.ApplyPatchType).
			AbsPath("/apis/apps/v1").
			Namespace(ns).
			Resource("deployments").
			Name("deployment-shared-map-item-removal").
			Param("fieldManager", "test_applier").
			Body(apply).
			Do(ctx).
			Get()
		if err != nil {
			framework.Failf("Failed to create object using Apply patch: %v", err)
		}

		// ensure the container is deleted even though a controller updated a field of the container
		deployment, ok := patched.(*appsv1.Deployment)
		if !ok {
			framework.Failf("Failed to convert response object to Deployment")
		}
		if *deployment.Spec.Replicas != 4 {
			framework.Failf("Expected deployment.spec.replicas to be 4, but got %d", deployment.Spec.Replicas)
		}
	})
})

// verifyNumFinalizers checks that len(.metadata.finalizers) == n
func verifyNumFinalizers(b []byte, n int) {
	obj := unstructured.Unstructured{}
	err := obj.UnmarshalJSON(b)
	if err != nil {
		framework.Failf("failed to unmarshal response: %v", err)
	}
	if actual, expected := len(obj.GetFinalizers()), n; actual != expected {
		framework.Failf("expected %v finalizers but got %v:\n%v", expected, actual, string(b))
	}
}

// verifyFinalizersIncludes checks that .metadata.finalizers includes e
func verifyFinalizersIncludes(b []byte, e string) {
	obj := unstructured.Unstructured{}
	err := obj.UnmarshalJSON(b)
	if err != nil {
		framework.Failf("failed to unmarshal response: %v", err)
	}
	for _, a := range obj.GetFinalizers() {
		if a == e {
			return
		}
	}
	framework.Failf("expected finalizers to include %q but got: %v", e, obj.GetFinalizers())
}

// verifyReplicas checks that .spec.replicas == r
func verifyReplicas(b []byte, r int) {
	obj := unstructured.Unstructured{}
	err := obj.UnmarshalJSON(b)
	if err != nil {
		framework.Failf("failed to find replicas number in response: %v:\n%v", err, string(b))
	}
	spec, ok := obj.Object["spec"]
	if !ok {
		framework.Failf("failed to find replicas number in response:\n%v", string(b))
	}
	specMap, ok := spec.(map[string]interface{})
	if !ok {
		framework.Failf("failed to find replicas number in response:\n%v", string(b))
	}
	replicas, ok := specMap["replicas"]
	if !ok {
		framework.Failf("failed to find replicas number in response:\n%v", string(b))
	}
	replicasNumber, ok := replicas.(int64)
	if !ok {
		framework.Failf("failed to find replicas number in response: expected int64 but got: %v", reflect.TypeOf(replicas))
	}
	if actual, expected := replicasNumber, int64(r); actual != expected {
		framework.Failf("expected %v ports but got %v:\n%v", expected, actual, string(b))
	}
}

// verifyNumPorts checks that len(.spec.ports) == n
func verifyNumPorts(b []byte, n int) {
	obj := unstructured.Unstructured{}
	err := obj.UnmarshalJSON(b)
	if err != nil {
		framework.Failf("failed to find ports list in response: %v:\n%v", err, string(b))
	}
	spec, ok := obj.Object["spec"]
	if !ok {
		framework.Failf("failed to find ports list in response:\n%v", string(b))
	}
	specMap, ok := spec.(map[string]interface{})
	if !ok {
		framework.Failf("failed to find ports list in response:\n%v", string(b))
	}
	ports, ok := specMap["ports"]
	if !ok {
		framework.Failf("failed to find ports list in response:\n%v", string(b))
	}
	portsList, ok := ports.([]interface{})
	if !ok {
		framework.Failf("failed to find ports list in response: expected array but got: %v", reflect.TypeOf(ports))
	}
	if actual, expected := len(portsList), n; actual != expected {
		framework.Failf("expected %v ports but got %v:\n%v", expected, actual, string(b))
	}
}

// verifyList checks that .spec.atomicList is the exact same as the expectedList provided
func verifyList(b []byte, expectedList []interface{}) {
	obj := unstructured.Unstructured{}
	err := obj.UnmarshalJSON(b)
	if err != nil {
		framework.Failf("failed to find atomicList in response: %v:\n%v", err, string(b))
	}
	spec, ok := obj.Object["spec"]
	if !ok {
		framework.Failf("failed to find atomicList in response:\n%v", string(b))
	}
	specMap, ok := spec.(map[string]interface{})
	if !ok {
		framework.Failf("failed to find atomicList in response:\n%v", string(b))
	}
	list, ok := specMap["atomicList"]
	if !ok {
		framework.Failf("failed to find atomicList in response:\n%v", string(b))
	}
	listString, ok := list.([]interface{})
	if !ok {
		framework.Failf("failed to find atomicList in response:\n%v", string(b))
	}
	if !reflect.DeepEqual(listString, expectedList) {
		framework.Failf("Expected list %s, got %s", expectedList, listString)
	}
}
