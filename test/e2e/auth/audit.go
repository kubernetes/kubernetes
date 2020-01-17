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

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/auth"
	e2edeploy "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	jsonpatch "github.com/evanphx/json-patch"
	"github.com/onsi/ginkgo"
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

// TODO: Get rid of [DisabledForLargeClusters] when feature request #53455 is ready.
// Marked as flaky until a reliable method for collecting server-side audit logs is available. See http://issue.k8s.io/74745#issuecomment-474052439
var _ = SIGDescribe("Advanced Audit [DisabledForLargeClusters][Flaky]", func() {
	f := framework.NewDefaultFramework("audit")
	var namespace string
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce")
		namespace = f.Namespace.Name
	})

	ginkgo.It("should audit API calls to create, get, update, patch, delete, list, watch pods.", func() {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "audit-pod",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{
					Name:  "pause",
					Image: imageutils.GetPauseImageName(),
				}},
			},
		}
		updatePod := func(pod *v1.Pod) {}

		f.PodClient().CreateSync(pod)

		_, err := f.PodClient().Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get audit-pod")

		podChan, err := f.PodClient().Watch(watchOptions)
		framework.ExpectNoError(err, "failed to create watch for pods")
		podChan.Stop()

		f.PodClient().Update(pod.Name, updatePod)

		_, err = f.PodClient().List(metav1.ListOptions{})
		framework.ExpectNoError(err, "failed to list pods")

		_, err = f.PodClient().Patch(pod.Name, types.JSONPatchType, patch)
		framework.ExpectNoError(err, "failed to patch pod")

		f.PodClient().DeleteSync(pod.Name, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)

		expectEvents(f, []utils.AuditEvent{
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
				RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/pods?timeout=%ds&timeoutSeconds=%d&watch=true", namespace, watchTestTimeout, watchTestTimeout),
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
				RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/pods?timeout=%ds&timeoutSeconds=%d&watch=true", namespace, watchTestTimeout, watchTestTimeout),
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
		})
	})

	ginkgo.It("should audit API calls to create, get, update, patch, delete, list, watch deployments.", func() {
		podLabels := map[string]string{"name": "audit-deployment-pod"}
		d := e2edeploy.NewDeployment("audit-deployment", int32(1), podLabels, "agnhost", imageutils.GetE2EImage(imageutils.Agnhost), appsv1.RecreateDeploymentStrategyType)

		_, err := f.ClientSet.AppsV1().Deployments(namespace).Create(d)
		framework.ExpectNoError(err, "failed to create audit-deployment")

		_, err = f.ClientSet.AppsV1().Deployments(namespace).Get(d.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get audit-deployment")

		deploymentChan, err := f.ClientSet.AppsV1().Deployments(namespace).Watch(watchOptions)
		framework.ExpectNoError(err, "failed to create watch for deployments")
		deploymentChan.Stop()

		_, err = f.ClientSet.AppsV1().Deployments(namespace).Update(d)
		framework.ExpectNoError(err, "failed to update audit-deployment")

		_, err = f.ClientSet.AppsV1().Deployments(namespace).Patch(d.Name, types.JSONPatchType, patch)
		framework.ExpectNoError(err, "failed to patch deployment")

		_, err = f.ClientSet.AppsV1().Deployments(namespace).List(metav1.ListOptions{})
		framework.ExpectNoError(err, "failed to create list deployments")

		err = f.ClientSet.AppsV1().Deployments(namespace).Delete("audit-deployment", &metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete deployments")

		expectEvents(f, []utils.AuditEvent{
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
				RequestURI:        fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments?timeout=%ds&timeoutSeconds=%d&watch=true", namespace, watchTestTimeout, watchTestTimeout),
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
				RequestURI:        fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments?timeout=%ds&timeoutSeconds=%d&watch=true", namespace, watchTestTimeout, watchTestTimeout),
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
		})
	})

	ginkgo.It("should audit API calls to create, get, update, patch, delete, list, watch configmaps.", func() {
		configMap := &v1.ConfigMap{
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
		configMapChan.Stop()

		_, err = f.ClientSet.CoreV1().ConfigMaps(namespace).Update(configMap)
		framework.ExpectNoError(err, "failed to update audit-configmap")

		_, err = f.ClientSet.CoreV1().ConfigMaps(namespace).Patch(configMap.Name, types.JSONPatchType, patch)
		framework.ExpectNoError(err, "failed to patch configmap")

		_, err = f.ClientSet.CoreV1().ConfigMaps(namespace).List(metav1.ListOptions{})
		framework.ExpectNoError(err, "failed to list config maps")

		err = f.ClientSet.CoreV1().ConfigMaps(namespace).Delete(configMap.Name, &metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete audit-configmap")

		expectEvents(f, []utils.AuditEvent{
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
				RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps?timeout=%ds&timeoutSeconds=%d&watch=true", namespace, watchTestTimeout, watchTestTimeout),
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
				RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps?timeout=%ds&timeoutSeconds=%d&watch=true", namespace, watchTestTimeout, watchTestTimeout),
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
		})
	})

	ginkgo.It("should audit API calls to create, get, update, patch, delete, list, watch secrets.", func() {
		secret := &v1.Secret{
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
		secretChan.Stop()

		_, err = f.ClientSet.CoreV1().Secrets(namespace).Update(secret)
		framework.ExpectNoError(err, "failed to update audit-secret")

		_, err = f.ClientSet.CoreV1().Secrets(namespace).Patch(secret.Name, types.JSONPatchType, patch)
		framework.ExpectNoError(err, "failed to patch secret")

		_, err = f.ClientSet.CoreV1().Secrets(namespace).List(metav1.ListOptions{})
		framework.ExpectNoError(err, "failed to list secrets")

		err = f.ClientSet.CoreV1().Secrets(namespace).Delete(secret.Name, &metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete audit-secret")

		expectEvents(f, []utils.AuditEvent{
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
				RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/secrets?timeout=%ds&timeoutSeconds=%d&watch=true", namespace, watchTestTimeout, watchTestTimeout),
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
				RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/secrets?timeout=%ds&timeoutSeconds=%d&watch=true", namespace, watchTestTimeout, watchTestTimeout),
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
		})
	})

	ginkgo.It("should audit API calls to create and delete custom resource definition.", func() {
		config, err := framework.LoadConfig()
		framework.ExpectNoError(err, "failed to load config")
		apiExtensionClient, err := apiextensionclientset.NewForConfig(config)
		framework.ExpectNoError(err, "failed to initialize apiExtensionClient")

		crd, err = fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionClient, f.DynamicClient)
		framework.ExpectNoError(err, "failed to create custom resource definition")
		err = fixtures.DeleteCustomResourceDefinition(crd, apiExtensionClient)
		framework.ExpectNoError(err, "failed to delete custom resource definition")

		expectEvents(f, []utils.AuditEvent{
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
		})
	})

	// test authorizer annotations, RBAC is required.
	ginkgo.It("should audit API calls to get a pod with unauthorized user.", func() {
		if !auth.IsRBACEnabled(f.ClientSet.RbacV1()) {
			e2eskipper.Skipf("RBAC not enabled.")
		}

		ginkgo.By("Creating a kubernetes client that impersonates an unauthorized anonymous user")
		config, err := framework.LoadConfig()
		framework.ExpectNoError(err)
		config.Impersonate = restclient.ImpersonationConfig{
			UserName: "system:anonymous",
			Groups:   []string{"system:unauthenticated"},
		}
		anonymousClient, err := clientset.NewForConfig(config)
		framework.ExpectNoError(err)

		_, err = anonymousClient.CoreV1().Pods(namespace).Get("another-audit-pod", metav1.GetOptions{})
		expectForbidden(err)

		expectEvents(f, []utils.AuditEvent{
			{
				Level:              auditinternal.LevelRequest,
				Stage:              auditinternal.StageResponseComplete,
				RequestURI:         fmt.Sprintf("/api/v1/namespaces/%s/pods/another-audit-pod", namespace),
				Verb:               "get",
				Code:               403,
				User:               auditTestUser,
				ImpersonatedUser:   "system:anonymous",
				ImpersonatedGroups: "system:unauthenticated",
				Resource:           "pods",
				Namespace:          namespace,
				RequestObject:      false,
				ResponseObject:     false,
				AuthorizeDecision:  "forbid",
			},
		})
	})

	ginkgo.It("should list pods as impersonated user.", func() {
		ginkgo.By("Creating a kubernetes client that impersonates an authorized user")
		config, err := framework.LoadConfig()
		framework.ExpectNoError(err)
		config.Impersonate = restclient.ImpersonationConfig{
			UserName: "superman",
			Groups:   []string{"system:masters"},
		}
		impersonatedClient, err := clientset.NewForConfig(config)
		framework.ExpectNoError(err)

		_, err = impersonatedClient.CoreV1().Pods(namespace).List(metav1.ListOptions{})
		framework.ExpectNoError(err, "failed to list pods")

		expectEvents(f, []utils.AuditEvent{
			{
				Level:              auditinternal.LevelRequest,
				Stage:              auditinternal.StageResponseComplete,
				RequestURI:         fmt.Sprintf("/api/v1/namespaces/%s/pods", namespace),
				Verb:               "list",
				Code:               200,
				User:               auditTestUser,
				ImpersonatedUser:   "superman",
				ImpersonatedGroups: "system:masters",
				Resource:           "pods",
				Namespace:          namespace,
				RequestObject:      false,
				ResponseObject:     false,
				AuthorizeDecision:  "allow",
			},
		})
	})

})

func expectEvents(f *framework.Framework, expectedEvents []utils.AuditEvent) {
	// The default flush timeout is 30 seconds, therefore it should be enough to retry once
	// to find all expected events. However, we're waiting for 5 minutes to avoid flakes.
	pollingInterval := 30 * time.Second
	pollingTimeout := 5 * time.Minute
	err := wait.Poll(pollingInterval, pollingTimeout, func() (bool, error) {
		// Fetch the log stream.
		stream, err := f.ClientSet.CoreV1().RESTClient().Get().AbsPath("/logs/kube-apiserver-audit.log").Stream()
		if err != nil {
			return false, err
		}
		defer stream.Close()
		missingReport, err := utils.CheckAuditLines(stream, expectedEvents, auditv1.SchemeGroupVersion)
		if err != nil {
			framework.Logf("Failed to observe audit events: %v", err)
		} else if len(missingReport.MissingEvents) > 0 {
			framework.Logf(missingReport.String())
		}
		return len(missingReport.MissingEvents) == 0, nil
	})
	framework.ExpectNoError(err, "after %v failed to observe audit events", pollingTimeout)
}
