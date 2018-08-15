/*
Copyright 2017 The Kubernetes Authors.

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

package auth

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	apps "k8s.io/api/apps/v1"
	apiv1 "k8s.io/api/core/v1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/apis/audit/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/evanphx/json-patch"
	. "github.com/onsi/ginkgo"
)

var (
	watchTestTimeout int64 = 1
	auditTestUser          = "kubecfg"

	crd          = fixtures.NewRandomNameCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
	crdName      = strings.SplitN(crd.Name, ".", 2)[0]
	crdNamespace = strings.SplitN(crd.Name, ".", 2)[1]

	watchOptions = metav1.ListOptions{TimeoutSeconds: &watchTestTimeout}
	patch, _     = json.Marshal(jsonpatch.Patch{})
)

var _ = SIGDescribe("Advanced Audit", func() {
	f := framework.NewDefaultFramework("audit")
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")
	})

	// TODO: Get rid of [DisabledForLargeClusters] when feature request #53455 is ready.
	It("should audit API calls [DisabledForLargeClusters]", func() {
		namespace := f.Namespace.Name

		config, err := framework.LoadConfig()
		framework.ExpectNoError(err, "failed to load config")
		apiExtensionClient, err := apiextensionclientset.NewForConfig(config)
		framework.ExpectNoError(err, "failed to initialize apiExtensionClient")

		By("Creating a kubernetes client that impersonates an unauthorized anonymous user")
		config, err = framework.LoadConfig()
		framework.ExpectNoError(err)
		config.Impersonate = restclient.ImpersonationConfig{
			UserName: "system:anonymous",
			Groups:   []string{"system:unauthenticated"},
		}
		anonymousClient, err := clientset.NewForConfig(config)
		framework.ExpectNoError(err)

		testCases := []struct {
			action func()
			events []utils.AuditEvent
		}{
			// Create, get, update, patch, delete, list, watch pods.
			{
				func() {
					pod := &apiv1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name: "audit-pod",
						},
						Spec: apiv1.PodSpec{
							Containers: []apiv1.Container{{
								Name:  "pause",
								Image: imageutils.GetPauseImageName(),
							}},
						},
					}
					updatePod := func(pod *apiv1.Pod) {}

					f.PodClient().CreateSync(pod)

					_, err := f.PodClient().Get(pod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err, "failed to get audit-pod")

					podChan, err := f.PodClient().Watch(watchOptions)
					framework.ExpectNoError(err, "failed to create watch for pods")
					for range podChan.ResultChan() {
					}

					f.PodClient().Update(pod.Name, updatePod)

					_, err = f.PodClient().List(metav1.ListOptions{})
					framework.ExpectNoError(err, "failed to list pods")

					_, err = f.PodClient().Patch(pod.Name, types.JSONPatchType, patch)
					framework.ExpectNoError(err, "failed to patch pod")

					f.PodClient().DeleteSync(pod.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)
				},
				[]utils.AuditEvent{
					{
						Level:             auditinternal.LevelRequestResponse,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/pods", namespace),
						Verb:              "create",
						Code:              201,
						User:              auditTestUser,
						Resource:          "pods",
						Namespace:         namespace,
						RequestObject:     true,
						ResponseObject:    true,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequest,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/pods/audit-pod", namespace),
						Verb:              "get",
						Code:              200,
						User:              auditTestUser,
						Resource:          "pods",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequest,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/pods", namespace),
						Verb:              "list",
						Code:              200,
						User:              auditTestUser,
						Resource:          "pods",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequest,
						Stage:             auditinternal.StageResponseStarted,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/pods?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						Verb:              "watch",
						Code:              200,
						User:              auditTestUser,
						Resource:          "pods",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequest,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/pods?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						Verb:              "watch",
						Code:              200,
						User:              auditTestUser,
						Resource:          "pods",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequestResponse,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/pods/audit-pod", namespace),
						Verb:              "update",
						Code:              200,
						User:              auditTestUser,
						Resource:          "pods",
						Namespace:         namespace,
						RequestObject:     true,
						ResponseObject:    true,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequestResponse,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/pods/audit-pod", namespace),
						Verb:              "patch",
						Code:              200,
						User:              auditTestUser,
						Resource:          "pods",
						Namespace:         namespace,
						RequestObject:     true,
						ResponseObject:    true,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequestResponse,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/pods/audit-pod", namespace),
						Verb:              "delete",
						Code:              200,
						User:              auditTestUser,
						Resource:          "pods",
						Namespace:         namespace,
						RequestObject:     true,
						ResponseObject:    true,
						AuthorizeDecision: "allow",
					},
				},
			},
			// Create, get, update, patch, delete, list, watch deployments.
			{
				func() {
					podLabels := map[string]string{"name": "audit-deployment-pod"}
					d := framework.NewDeployment("audit-deployment", int32(1), podLabels, "redis", imageutils.GetE2EImage(imageutils.Redis), apps.RecreateDeploymentStrategyType)

					_, err := f.ClientSet.AppsV1().Deployments(namespace).Create(d)
					framework.ExpectNoError(err, "failed to create audit-deployment")

					_, err = f.ClientSet.AppsV1().Deployments(namespace).Get(d.Name, metav1.GetOptions{})
					framework.ExpectNoError(err, "failed to get audit-deployment")

					deploymentChan, err := f.ClientSet.AppsV1().Deployments(namespace).Watch(watchOptions)
					framework.ExpectNoError(err, "failed to create watch for deployments")
					for range deploymentChan.ResultChan() {
					}

					_, err = f.ClientSet.AppsV1().Deployments(namespace).Update(d)
					framework.ExpectNoError(err, "failed to update audit-deployment")

					_, err = f.ClientSet.AppsV1().Deployments(namespace).Patch(d.Name, types.JSONPatchType, patch)
					framework.ExpectNoError(err, "failed to patch deployment")

					_, err = f.ClientSet.AppsV1().Deployments(namespace).List(metav1.ListOptions{})
					framework.ExpectNoError(err, "failed to create list deployments")

					err = f.ClientSet.AppsV1().Deployments(namespace).Delete("audit-deployment", &metav1.DeleteOptions{})
					framework.ExpectNoError(err, "failed to delete deployments")
				},
				[]utils.AuditEvent{
					{
						Level:             auditinternal.LevelRequestResponse,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments", namespace),
						Verb:              "create",
						Code:              201,
						User:              auditTestUser,
						Resource:          "deployments",
						Namespace:         namespace,
						RequestObject:     true,
						ResponseObject:    true,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequest,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments/audit-deployment", namespace),
						Verb:              "get",
						Code:              200,
						User:              auditTestUser,
						Resource:          "deployments",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequest,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments", namespace),
						Verb:              "list",
						Code:              200,
						User:              auditTestUser,
						Resource:          "deployments",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequest,
						Stage:             auditinternal.StageResponseStarted,
						RequestURI:        fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						Verb:              "watch",
						Code:              200,
						User:              auditTestUser,
						Resource:          "deployments",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequest,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						Verb:              "watch",
						Code:              200,
						User:              auditTestUser,
						Resource:          "deployments",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequestResponse,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments/audit-deployment", namespace),
						Verb:              "update",
						Code:              200,
						User:              auditTestUser,
						Resource:          "deployments",
						Namespace:         namespace,
						RequestObject:     true,
						ResponseObject:    true,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequestResponse,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments/audit-deployment", namespace),
						Verb:              "patch",
						Code:              200,
						User:              auditTestUser,
						Resource:          "deployments",
						Namespace:         namespace,
						RequestObject:     true,
						ResponseObject:    true,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequestResponse,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments/audit-deployment", namespace),
						Verb:              "delete",
						Code:              200,
						User:              auditTestUser,
						Resource:          "deployments",
						Namespace:         namespace,
						RequestObject:     true,
						ResponseObject:    true,
						AuthorizeDecision: "allow",
					},
				},
			},
			// Create, get, update, patch, delete, list, watch configmaps.
			{
				func() {
					configMap := &apiv1.ConfigMap{
						ObjectMeta: metav1.ObjectMeta{
							Name: "audit-configmap",
						},
						Data: map[string]string{
							"map-key": "map-value",
						},
					}

					_, err := f.ClientSet.CoreV1().ConfigMaps(namespace).Create(configMap)
					framework.ExpectNoError(err, "failed to create audit-configmap")

					_, err = f.ClientSet.CoreV1().ConfigMaps(namespace).Get(configMap.Name, metav1.GetOptions{})
					framework.ExpectNoError(err, "failed to get audit-configmap")

					configMapChan, err := f.ClientSet.CoreV1().ConfigMaps(namespace).Watch(watchOptions)
					framework.ExpectNoError(err, "failed to create watch for config maps")
					for range configMapChan.ResultChan() {
					}

					_, err = f.ClientSet.CoreV1().ConfigMaps(namespace).Update(configMap)
					framework.ExpectNoError(err, "failed to update audit-configmap")

					_, err = f.ClientSet.CoreV1().ConfigMaps(namespace).Patch(configMap.Name, types.JSONPatchType, patch)
					framework.ExpectNoError(err, "failed to patch configmap")

					_, err = f.ClientSet.CoreV1().ConfigMaps(namespace).List(metav1.ListOptions{})
					framework.ExpectNoError(err, "failed to list config maps")

					err = f.ClientSet.CoreV1().ConfigMaps(namespace).Delete(configMap.Name, &metav1.DeleteOptions{})
					framework.ExpectNoError(err, "failed to delete audit-configmap")
				},
				[]utils.AuditEvent{
					{
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps", namespace),
						Verb:              "create",
						Code:              201,
						User:              auditTestUser,
						Resource:          "configmaps",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps/audit-configmap", namespace),
						Verb:              "get",
						Code:              200,
						User:              auditTestUser,
						Resource:          "configmaps",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps", namespace),
						Verb:              "list",
						Code:              200,
						User:              auditTestUser,
						Resource:          "configmaps",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseStarted,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						Verb:              "watch",
						Code:              200,
						User:              auditTestUser,
						Resource:          "configmaps",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						Verb:              "watch",
						Code:              200,
						User:              auditTestUser,
						Resource:          "configmaps",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps/audit-configmap", namespace),
						Verb:              "update",
						Code:              200,
						User:              auditTestUser,
						Resource:          "configmaps",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps/audit-configmap", namespace),
						Verb:              "patch",
						Code:              200,
						User:              auditTestUser,
						Resource:          "configmaps",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps/audit-configmap", namespace),
						Verb:              "delete",
						Code:              200,
						User:              auditTestUser,
						Resource:          "configmaps",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					},
				},
			},
			// Create, get, update, patch, delete, list, watch secrets.
			{
				func() {
					secret := &apiv1.Secret{
						ObjectMeta: metav1.ObjectMeta{
							Name: "audit-secret",
						},
						Data: map[string][]byte{
							"top-secret": []byte("foo-bar"),
						},
					}
					_, err := f.ClientSet.CoreV1().Secrets(namespace).Create(secret)
					framework.ExpectNoError(err, "failed to create audit-secret")

					_, err = f.ClientSet.CoreV1().Secrets(namespace).Get(secret.Name, metav1.GetOptions{})
					framework.ExpectNoError(err, "failed to get audit-secret")

					secretChan, err := f.ClientSet.CoreV1().Secrets(namespace).Watch(watchOptions)
					framework.ExpectNoError(err, "failed to create watch for secrets")
					for range secretChan.ResultChan() {
					}

					_, err = f.ClientSet.CoreV1().Secrets(namespace).Update(secret)
					framework.ExpectNoError(err, "failed to update audit-secret")

					_, err = f.ClientSet.CoreV1().Secrets(namespace).Patch(secret.Name, types.JSONPatchType, patch)
					framework.ExpectNoError(err, "failed to patch secret")

					_, err = f.ClientSet.CoreV1().Secrets(namespace).List(metav1.ListOptions{})
					framework.ExpectNoError(err, "failed to list secrets")

					err = f.ClientSet.CoreV1().Secrets(namespace).Delete(secret.Name, &metav1.DeleteOptions{})
					framework.ExpectNoError(err, "failed to delete audit-secret")
				},
				[]utils.AuditEvent{
					{
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/secrets", namespace),
						Verb:              "create",
						Code:              201,
						User:              auditTestUser,
						Resource:          "secrets",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/secrets/audit-secret", namespace),
						Verb:              "get",
						Code:              200,
						User:              auditTestUser,
						Resource:          "secrets",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/secrets", namespace),
						Verb:              "list",
						Code:              200,
						User:              auditTestUser,
						Resource:          "secrets",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseStarted,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/secrets?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						Verb:              "watch",
						Code:              200,
						User:              auditTestUser,
						Resource:          "secrets",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/secrets?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						Verb:              "watch",
						Code:              200,
						User:              auditTestUser,
						Resource:          "secrets",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/secrets/audit-secret", namespace),
						Verb:              "update",
						Code:              200,
						User:              auditTestUser,
						Resource:          "secrets",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/secrets/audit-secret", namespace),
						Verb:              "patch",
						Code:              200,
						User:              auditTestUser,
						Resource:          "secrets",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/secrets/audit-secret", namespace),
						Verb:              "delete",
						Code:              200,
						User:              auditTestUser,
						Resource:          "secrets",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					},
				},
			},
			// Create and delete custom resource definition.
			{
				func() {
					crd, err = fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionClient, f.DynamicClient)
					framework.ExpectNoError(err, "failed to create custom resource definition")
					fixtures.DeleteCustomResourceDefinition(crd, apiExtensionClient)
				},
				[]utils.AuditEvent{
					{
						Level:             auditinternal.LevelRequestResponse,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        "/apis/apiextensions.k8s.io/v1beta1/customresourcedefinitions",
						Verb:              "create",
						Code:              201,
						User:              auditTestUser,
						Resource:          "customresourcedefinitions",
						RequestObject:     true,
						ResponseObject:    true,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/apis/%s/v1beta1/%s", crdNamespace, crdName),
						Verb:              "create",
						Code:              201,
						User:              auditTestUser,
						Resource:          crdName,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelRequestResponse,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/apis/apiextensions.k8s.io/v1beta1/customresourcedefinitions/%s", crd.Name),
						Verb:              "delete",
						Code:              200,
						User:              auditTestUser,
						Resource:          "customresourcedefinitions",
						RequestObject:     false,
						ResponseObject:    true,
						AuthorizeDecision: "allow",
					}, {
						Level:             auditinternal.LevelMetadata,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/apis/%s/v1beta1/%s/setup-instance", crdNamespace, crdName),
						Verb:              "delete",
						Code:              200,
						User:              auditTestUser,
						Resource:          crdName,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "allow",
					},
				},
			},
		}

		// test authorizer annotations, RBAC is required.
		annotationTestCases := []struct {
			action func()
			events []utils.AuditEvent
		}{

			// get a pod with unauthorized user
			{
				func() {
					_, err := anonymousClient.CoreV1().Pods(namespace).Get("another-audit-pod", metav1.GetOptions{})
					expectForbidden(err)
				},
				[]utils.AuditEvent{
					{
						Level:             auditinternal.LevelRequest,
						Stage:             auditinternal.StageResponseComplete,
						RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/pods/another-audit-pod", namespace),
						Verb:              "get",
						Code:              403,
						User:              auditTestUser,
						Resource:          "pods",
						Namespace:         namespace,
						RequestObject:     false,
						ResponseObject:    false,
						AuthorizeDecision: "forbid",
					},
				},
			},
		}

		if framework.IsRBACEnabled(f) {
			testCases = append(testCases, annotationTestCases...)
		}
		expectedEvents := []utils.AuditEvent{}
		for _, t := range testCases {
			t.action()
			expectedEvents = append(expectedEvents, t.events...)
		}

		// The default flush timeout is 30 seconds, therefore it should be enough to retry once
		// to find all expected events. However, we're waiting for 5 minutes to avoid flakes.
		pollingInterval := 30 * time.Second
		pollingTimeout := 5 * time.Minute
		err = wait.Poll(pollingInterval, pollingTimeout, func() (bool, error) {
			// Fetch the log stream.
			stream, err := f.ClientSet.CoreV1().RESTClient().Get().AbsPath("/logs/kube-apiserver-audit.log").Stream()
			if err != nil {
				return false, err
			}
			defer stream.Close()
			missing, err := utils.CheckAuditLines(stream, expectedEvents, v1beta1.SchemeGroupVersion)
			if err != nil {
				framework.Logf("Failed to observe audit events: %v", err)
			} else if len(missing) > 0 {
				framework.Logf("Events %#v not found!", missing)
			}
			return len(missing) == 0, nil
		})
		framework.ExpectNoError(err, "after %v failed to observe audit events", pollingTimeout)
	})
})
