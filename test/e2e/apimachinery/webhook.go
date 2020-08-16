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

package apimachinery

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"time"

	"k8s.io/utils/pointer"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	crdclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/crd"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	// ensure libs have a chance to initialize
	_ "github.com/stretchr/testify/assert"
)

const (
	secretName      = "sample-webhook-secret"
	deploymentName  = "sample-webhook-deployment"
	serviceName     = "e2e-test-webhook"
	roleBindingName = "webhook-auth-reader"

	skipNamespaceLabelKey   = "skip-webhook-admission"
	skipNamespaceLabelValue = "yes"
	skippedNamespaceName    = "exempted-namesapce"
	disallowedPodName       = "disallowed-pod"
	toBeAttachedPodName     = "to-be-attached-pod"
	hangingPodName          = "hanging-pod"
	disallowedConfigMapName = "disallowed-configmap"
	allowedConfigMapName    = "allowed-configmap"
	failNamespaceLabelKey   = "fail-closed-webhook"
	failNamespaceLabelValue = "yes"
	failNamespaceName       = "fail-closed-namesapce"
	addedLabelKey           = "added-label"
	addedLabelValue         = "yes"
)

var _ = SIGDescribe("AdmissionWebhook [Privileged:ClusterAdmin]", func() {
	var certCtx *certContext
	f := framework.NewDefaultFramework("webhook")
	servicePort := int32(8443)
	containerPort := int32(8444)

	var client clientset.Interface
	var namespaceName string

	ginkgo.BeforeEach(func() {
		client = f.ClientSet
		namespaceName = f.Namespace.Name

		// Make sure the namespace created for the test is labeled to be selected by the webhooks
		labelNamespace(f, f.Namespace.Name)
		createWebhookConfigurationReadyNamespace(f)

		ginkgo.By("Setting up server cert")
		certCtx = setupServerCert(namespaceName, serviceName)
		createAuthReaderRoleBinding(f, namespaceName)

		deployWebhookAndService(f, imageutils.GetE2EImage(imageutils.Agnhost), certCtx, servicePort, containerPort)
	})

	ginkgo.AfterEach(func() {
		cleanWebhookTest(client, namespaceName)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, discovery document
		Description: The admissionregistration.k8s.io API group MUST exists in the /apis discovery document.
		The admissionregistration.k8s.io/v1 API group/version MUST exists in the /apis discovery document.
		The mutatingwebhookconfigurations and validatingwebhookconfigurations resources MUST exist in the
		/apis/admissionregistration.k8s.io/v1 discovery document.
	*/
	framework.ConformanceIt("should include webhook resources in discovery documents", func() {
		{
			ginkgo.By("fetching the /apis discovery document")
			apiGroupList := &metav1.APIGroupList{}
			err := client.Discovery().RESTClient().Get().AbsPath("/apis").Do(context.TODO()).Into(apiGroupList)
			framework.ExpectNoError(err, "fetching /apis")

			ginkgo.By("finding the admissionregistration.k8s.io API group in the /apis discovery document")
			var group *metav1.APIGroup
			for _, g := range apiGroupList.Groups {
				if g.Name == admissionregistrationv1.GroupName {
					group = &g
					break
				}
			}
			framework.ExpectNotEqual(group, nil, "admissionregistration.k8s.io API group not found in /apis discovery document")

			ginkgo.By("finding the admissionregistration.k8s.io/v1 API group/version in the /apis discovery document")
			var version *metav1.GroupVersionForDiscovery
			for _, v := range group.Versions {
				if v.Version == admissionregistrationv1.SchemeGroupVersion.Version {
					version = &v
					break
				}
			}
			framework.ExpectNotEqual(version, nil, "admissionregistration.k8s.io/v1 API group version not found in /apis discovery document")
		}

		{
			ginkgo.By("fetching the /apis/admissionregistration.k8s.io discovery document")
			group := &metav1.APIGroup{}
			err := client.Discovery().RESTClient().Get().AbsPath("/apis/admissionregistration.k8s.io").Do(context.TODO()).Into(group)
			framework.ExpectNoError(err, "fetching /apis/admissionregistration.k8s.io")
			framework.ExpectEqual(group.Name, admissionregistrationv1.GroupName, "verifying API group name in /apis/admissionregistration.k8s.io discovery document")

			ginkgo.By("finding the admissionregistration.k8s.io/v1 API group/version in the /apis/admissionregistration.k8s.io discovery document")
			var version *metav1.GroupVersionForDiscovery
			for _, v := range group.Versions {
				if v.Version == admissionregistrationv1.SchemeGroupVersion.Version {
					version = &v
					break
				}
			}
			framework.ExpectNotEqual(version, nil, "admissionregistration.k8s.io/v1 API group version not found in /apis/admissionregistration.k8s.io discovery document")
		}

		{
			ginkgo.By("fetching the /apis/admissionregistration.k8s.io/v1 discovery document")
			apiResourceList := &metav1.APIResourceList{}
			err := client.Discovery().RESTClient().Get().AbsPath("/apis/admissionregistration.k8s.io/v1").Do(context.TODO()).Into(apiResourceList)
			framework.ExpectNoError(err, "fetching /apis/admissionregistration.k8s.io/v1")
			framework.ExpectEqual(apiResourceList.GroupVersion, admissionregistrationv1.SchemeGroupVersion.String(), "verifying API group/version in /apis/admissionregistration.k8s.io/v1 discovery document")

			ginkgo.By("finding mutatingwebhookconfigurations and validatingwebhookconfigurations resources in the /apis/admissionregistration.k8s.io/v1 discovery document")
			var (
				mutatingWebhookResource   *metav1.APIResource
				validatingWebhookResource *metav1.APIResource
			)
			for i := range apiResourceList.APIResources {
				if apiResourceList.APIResources[i].Name == "mutatingwebhookconfigurations" {
					mutatingWebhookResource = &apiResourceList.APIResources[i]
				}
				if apiResourceList.APIResources[i].Name == "validatingwebhookconfigurations" {
					validatingWebhookResource = &apiResourceList.APIResources[i]
				}
			}
			framework.ExpectNotEqual(mutatingWebhookResource, nil, "mutatingwebhookconfigurations resource not found in /apis/admissionregistration.k8s.io/v1 discovery document")
			framework.ExpectNotEqual(validatingWebhookResource, nil, "validatingwebhookconfigurations resource not found in /apis/admissionregistration.k8s.io/v1 discovery document")
		}
	})

	/*
		Release: v1.16
		Testname: Admission webhook, deny create
		Description: Register an admission webhook configuration that admits pod and configmap. Attempts to create
		non-compliant pods and configmaps, or update/patch compliant pods and configmaps to be non-compliant MUST
		be denied. An attempt to create a pod that causes a webhook to hang MUST result in a webhook timeout error,
		and the pod creation MUST be denied. An attempt to create a non-compliant configmap in a whitelisted
		namespace based on the webhook namespace selector MUST be allowed.
	*/
	framework.ConformanceIt("should be able to deny pod and configmap creation", func() {
		webhookCleanup := registerWebhook(f, f.UniqueName, certCtx, servicePort)
		defer webhookCleanup()
		testWebhook(f)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, deny attach
		Description: Register an admission webhook configuration that denies connecting to a pod's attach sub-resource.
		Attempts to attach MUST be denied.
	*/
	framework.ConformanceIt("should be able to deny attaching pod", func() {
		webhookCleanup := registerWebhookForAttachingPod(f, f.UniqueName, certCtx, servicePort)
		defer webhookCleanup()
		testAttachingPodWebhook(f)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, deny custom resource create and delete
		Description: Register an admission webhook configuration that denies creation, update and deletion of
		custom resources. Attempts to create, update and delete custom resources MUST be denied.
	*/
	framework.ConformanceIt("should be able to deny custom resource creation, update and deletion", func() {
		testcrd, err := crd.CreateTestCRD(f)
		if err != nil {
			return
		}
		defer testcrd.CleanUp()
		webhookCleanup := registerWebhookForCustomResource(f, f.UniqueName, certCtx, testcrd, servicePort)
		defer webhookCleanup()
		testCustomResourceWebhook(f, testcrd.Crd, testcrd.DynamicClients["v1"])
		testBlockingCustomResourceUpdateDeletion(f, testcrd.Crd, testcrd.DynamicClients["v1"])
	})

	/*
		Release: v1.16
		Testname: Admission webhook, fail closed
		Description: Register a webhook with a fail closed policy and without CA bundle so that it cannot be called.
		Attempt operations that require the admission webhook; all MUST be denied.
	*/
	framework.ConformanceIt("should unconditionally reject operations on fail closed webhook", func() {
		webhookCleanup := registerFailClosedWebhook(f, f.UniqueName, certCtx, servicePort)
		defer webhookCleanup()
		testFailClosedWebhook(f)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, ordered mutation
		Description: Register a mutating webhook configuration with two webhooks that admit configmaps, one that
		adds a data key if the configmap already has a specific key, and another that adds a key if the key added by
		the first webhook is present. Attempt to create a config map; both keys MUST be added to the config map.
	*/
	framework.ConformanceIt("should mutate configmap", func() {
		webhookCleanup := registerMutatingWebhookForConfigMap(f, f.UniqueName, certCtx, servicePort)
		defer webhookCleanup()
		testMutatingConfigMapWebhook(f)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, mutation with defaulting
		Description: Register a mutating webhook that adds an InitContainer to pods. Attempt to create a pod;
		the InitContainer MUST be added the TerminationMessagePolicy MUST be defaulted.
	*/
	framework.ConformanceIt("should mutate pod and apply defaults after mutation", func() {
		webhookCleanup := registerMutatingWebhookForPod(f, f.UniqueName, certCtx, servicePort)
		defer webhookCleanup()
		testMutatingPodWebhook(f)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, admission control not allowed on webhook configuration objects
		Description: Register webhooks that mutate and deny deletion of webhook configuration objects. Attempt to create
		and delete a webhook configuration object; both operations MUST be allowed and the webhook configuration object
		MUST NOT be mutated the webhooks.
	*/
	framework.ConformanceIt("should not be able to mutate or prevent deletion of webhook configuration objects", func() {
		validatingWebhookCleanup := registerValidatingWebhookForWebhookConfigurations(f, f.UniqueName+"blocking", certCtx, servicePort)
		defer validatingWebhookCleanup()
		mutatingWebhookCleanup := registerMutatingWebhookForWebhookConfigurations(f, f.UniqueName+"blocking", certCtx, servicePort)
		defer mutatingWebhookCleanup()
		testWebhooksForWebhookConfigurations(f, f.UniqueName, certCtx, servicePort)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, mutate custom resource
		Description: Register a webhook that mutates a custom resource. Attempt to create custom resource object;
		the custom resource MUST be mutated.
	*/
	framework.ConformanceIt("should mutate custom resource", func() {
		testcrd, err := crd.CreateTestCRD(f)
		if err != nil {
			return
		}
		defer testcrd.CleanUp()
		webhookCleanup := registerMutatingWebhookForCustomResource(f, f.UniqueName, certCtx, testcrd, servicePort)
		defer webhookCleanup()
		testMutatingCustomResourceWebhook(f, testcrd.Crd, testcrd.DynamicClients["v1"], false)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, deny custom resource definition
		Description: Register a webhook that denies custom resource definition create. Attempt to create a
		custom resource definition; the create request MUST be denied.
	*/
	framework.ConformanceIt("should deny crd creation", func() {
		crdWebhookCleanup := registerValidatingWebhookForCRD(f, f.UniqueName, certCtx, servicePort)
		defer crdWebhookCleanup()

		testCRDDenyWebhook(f)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, mutate custom resource with different stored version
		Description: Register a webhook that mutates custom resources on create and update. Register a custom resource
		definition using v1 as stored version. Create a custom resource. Patch the custom resource definition to use v2 as
		the stored version. Attempt to patch the custom resource with a new field and value; the patch MUST be applied
		successfully.
	*/
	framework.ConformanceIt("should mutate custom resource with different stored version", func() {
		testcrd, err := createAdmissionWebhookMultiVersionTestCRDWithV1Storage(f)
		if err != nil {
			return
		}
		defer testcrd.CleanUp()
		webhookCleanup := registerMutatingWebhookForCustomResource(f, f.UniqueName, certCtx, testcrd, servicePort)
		defer webhookCleanup()
		testMultiVersionCustomResourceWebhook(f, testcrd)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, mutate custom resource with pruning
		Description: Register mutating webhooks that adds fields to custom objects. Register a custom resource definition
		with a schema that includes only one of the data keys added by the webhooks. Attempt to a custom resource;
		the fields included in the schema MUST be present and field not included in the schema MUST NOT be present.
	*/
	framework.ConformanceIt("should mutate custom resource with pruning", func() {
		const prune = true
		testcrd, err := createAdmissionWebhookMultiVersionTestCRDWithV1Storage(f, func(crd *apiextensionsv1.CustomResourceDefinition) {
			crd.Spec.PreserveUnknownFields = false
			for i := range crd.Spec.Versions {
				crd.Spec.Versions[i].Schema = &apiextensionsv1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
						Type: "object",
						Properties: map[string]apiextensionsv1.JSONSchemaProps{
							"data": {
								Type: "object",
								Properties: map[string]apiextensionsv1.JSONSchemaProps{
									"mutation-start":   {Type: "string"},
									"mutation-stage-1": {Type: "string"},
									// mutation-stage-2 is intentionally missing such that it is pruned
								},
							},
						},
					},
				}
			}
		})
		if err != nil {
			return
		}
		defer testcrd.CleanUp()
		webhookCleanup := registerMutatingWebhookForCustomResource(f, f.UniqueName, certCtx, testcrd, servicePort)
		defer webhookCleanup()
		testMutatingCustomResourceWebhook(f, testcrd.Crd, testcrd.DynamicClients["v1"], prune)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, honor timeout
		Description: Using a webhook that waits 5 seconds before admitting objects, configure the webhook with combinations
		of timeouts and failure policy values. Attempt to create a config map with each combination. Requests MUST
		timeout if the configured webhook timeout is less than 5 seconds and failure policy is fail. Requests must not timeout if
		the failure policy is ignore. Requests MUST NOT timeout if configured webhook timeout is 10 seconds (much longer
		than the webhook wait duration).
	*/
	framework.ConformanceIt("should honor timeout", func() {
		policyFail := admissionregistrationv1.Fail
		policyIgnore := admissionregistrationv1.Ignore

		ginkgo.By("Setting timeout (1s) shorter than webhook latency (5s)")
		slowWebhookCleanup := registerSlowWebhook(f, f.UniqueName, certCtx, &policyFail, pointer.Int32Ptr(1), servicePort)
		testSlowWebhookTimeoutFailEarly(f)
		slowWebhookCleanup()

		ginkgo.By("Having no error when timeout is shorter than webhook latency and failure policy is ignore")
		slowWebhookCleanup = registerSlowWebhook(f, f.UniqueName, certCtx, &policyIgnore, pointer.Int32Ptr(1), servicePort)
		testSlowWebhookTimeoutNoError(f)
		slowWebhookCleanup()

		ginkgo.By("Having no error when timeout is longer than webhook latency")
		slowWebhookCleanup = registerSlowWebhook(f, f.UniqueName, certCtx, &policyFail, pointer.Int32Ptr(10), servicePort)
		testSlowWebhookTimeoutNoError(f)
		slowWebhookCleanup()

		ginkgo.By("Having no error when timeout is empty (defaulted to 10s in v1)")
		slowWebhookCleanup = registerSlowWebhook(f, f.UniqueName, certCtx, &policyFail, nil, servicePort)
		testSlowWebhookTimeoutNoError(f)
		slowWebhookCleanup()
	})

	/*
		Release: v1.16
		Testname: Admission webhook, update validating webhook
		Description: Register a validating admission webhook configuration. Update the webhook to not apply to the create
		operation and attempt to create an object; the webhook MUST NOT deny the create. Patch the webhook to apply to the
		create operation again and attempt to create an object; the webhook MUST deny the create.
	*/
	framework.ConformanceIt("patching/updating a validating webhook should work", func() {
		client := f.ClientSet
		admissionClient := client.AdmissionregistrationV1()

		ginkgo.By("Creating a validating webhook configuration")
		hook, err := createValidatingWebhookConfiguration(f, &admissionregistrationv1.ValidatingWebhookConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				Name: f.UniqueName,
			},
			Webhooks: []admissionregistrationv1.ValidatingWebhook{
				newDenyConfigMapWebhookFixture(f, certCtx, servicePort),
				newValidatingIsReadyWebhookFixture(f, certCtx, servicePort),
			},
		})
		framework.ExpectNoError(err, "Creating validating webhook configuration")
		defer func() {
			err := client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(context.TODO(), hook.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Deleting validating webhook configuration")
		}()

		// ensure backend is ready before proceeding
		err = waitWebhookConfigurationReady(f)
		framework.ExpectNoError(err, "waiting for webhook configuration to be ready")

		ginkgo.By("Creating a configMap that does not comply to the validation webhook rules")
		err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
			cm := namedNonCompliantConfigMap(string(uuid.NewUUID()), f)
			_, err = client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), cm, metav1.CreateOptions{})
			if err == nil {
				err = client.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), cm.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, "Deleting successfully created configMap")
				return false, nil
			}
			if !strings.Contains(err.Error(), "denied") {
				return false, err
			}
			return true, nil
		})

		ginkgo.By("Updating a validating webhook configuration's rules to not include the create operation")
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			h, err := admissionClient.ValidatingWebhookConfigurations().Get(context.TODO(), f.UniqueName, metav1.GetOptions{})
			framework.ExpectNoError(err, "Getting validating webhook configuration")
			h.Webhooks[0].Rules[0].Operations = []admissionregistrationv1.OperationType{admissionregistrationv1.Update}
			_, err = admissionClient.ValidatingWebhookConfigurations().Update(context.TODO(), h, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "Updating validating webhook configuration")

		ginkgo.By("Creating a configMap that does not comply to the validation webhook rules")
		err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
			cm := namedNonCompliantConfigMap(string(uuid.NewUUID()), f)
			_, err = client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), cm, metav1.CreateOptions{})
			if err != nil {
				if !strings.Contains(err.Error(), "denied") {
					return false, err
				}
				return false, nil
			}
			err = client.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), cm.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Deleting successfully created configMap")
			return true, nil
		})
		framework.ExpectNoError(err, "Waiting for configMap in namespace %s to be allowed creation since webhook was updated to not validate create", f.Namespace.Name)

		ginkgo.By("Patching a validating webhook configuration's rules to include the create operation")
		hook, err = admissionClient.ValidatingWebhookConfigurations().Patch(context.TODO(), f.UniqueName,
			types.JSONPatchType,
			[]byte(`[{"op": "replace", "path": "/webhooks/0/rules/0/operations", "value": ["CREATE"]}]`), metav1.PatchOptions{})
		framework.ExpectNoError(err, "Patching validating webhook configuration")

		ginkgo.By("Creating a configMap that does not comply to the validation webhook rules")
		err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
			cm := namedNonCompliantConfigMap(string(uuid.NewUUID()), f)
			_, err = client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), cm, metav1.CreateOptions{})
			if err == nil {
				err = client.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), cm.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, "Deleting successfully created configMap")
				return false, nil
			}
			if !strings.Contains(err.Error(), "denied") {
				return false, err
			}
			return true, nil
		})
		framework.ExpectNoError(err, "Waiting for configMap in namespace %s to be denied creation by validating webhook", f.Namespace.Name)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, update mutating webhook
		Description: Register a mutating admission webhook configuration. Update the webhook to not apply to the create
		operation and attempt to create an object; the webhook MUST NOT mutate the object. Patch the webhook to apply to the
		create operation again and attempt to create an object; the webhook MUST mutate the object.
	*/
	framework.ConformanceIt("patching/updating a mutating webhook should work", func() {
		client := f.ClientSet
		admissionClient := client.AdmissionregistrationV1()

		ginkgo.By("Creating a mutating webhook configuration")
		hook, err := createMutatingWebhookConfiguration(f, &admissionregistrationv1.MutatingWebhookConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				Name: f.UniqueName,
			},
			Webhooks: []admissionregistrationv1.MutatingWebhook{
				newMutateConfigMapWebhookFixture(f, certCtx, 1, servicePort),
				newMutatingIsReadyWebhookFixture(f, certCtx, servicePort),
			},
		})
		framework.ExpectNoError(err, "Creating mutating webhook configuration")
		defer func() {
			err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(context.TODO(), hook.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Deleting mutating webhook configuration")
		}()

		// ensure backend is ready before proceeding
		err = waitWebhookConfigurationReady(f)
		framework.ExpectNoError(err, "waiting for webhook configuration to be ready")

		hook, err = admissionClient.MutatingWebhookConfigurations().Get(context.TODO(), f.UniqueName, metav1.GetOptions{})
		framework.ExpectNoError(err, "Getting mutating webhook configuration")
		ginkgo.By("Updating a mutating webhook configuration's rules to not include the create operation")
		hook.Webhooks[0].Rules[0].Operations = []admissionregistrationv1.OperationType{admissionregistrationv1.Update}
		hook, err = admissionClient.MutatingWebhookConfigurations().Update(context.TODO(), hook, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "Updating mutating webhook configuration")

		ginkgo.By("Creating a configMap that should not be mutated")
		err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
			cm := namedToBeMutatedConfigMap(string(uuid.NewUUID()), f)
			created, err := client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), cm, metav1.CreateOptions{})
			if err != nil {
				return false, err
			}
			err = client.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), cm.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Deleting successfully created configMap")
			_, ok := created.Data["mutation-stage-1"]
			return !ok, nil
		})
		framework.ExpectNoError(err, "Waiting for configMap in namespace %s this is not mutated", f.Namespace.Name)

		ginkgo.By("Patching a mutating webhook configuration's rules to include the create operation")
		hook, err = admissionClient.MutatingWebhookConfigurations().Patch(context.TODO(), f.UniqueName,
			types.JSONPatchType,
			[]byte(`[{"op": "replace", "path": "/webhooks/0/rules/0/operations", "value": ["CREATE"]}]`), metav1.PatchOptions{})
		framework.ExpectNoError(err, "Patching mutating webhook configuration")

		ginkgo.By("Creating a configMap that should be mutated")
		err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
			cm := namedToBeMutatedConfigMap(string(uuid.NewUUID()), f)
			created, err := client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), cm, metav1.CreateOptions{})
			if err != nil {
				return false, err
			}
			err = client.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), cm.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Deleting successfully created configMap")
			_, ok := created.Data["mutation-stage-1"]
			return ok, nil
		})
		framework.ExpectNoError(err, "Waiting for configMap in namespace %s to be mutated", f.Namespace.Name)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, list validating webhooks
		Description: Create 10 validating webhook configurations, all with a label. Attempt to list the webhook
		configurations matching the label; all the created webhook configurations MUST be present. Attempt to create an
		object; the create MUST be denied. Attempt to remove the webhook configurations matching the label with deletecollection;
		all webhook configurations MUST be deleted. Attempt to create an object; the create MUST NOT be denied.
	*/
	framework.ConformanceIt("listing validating webhooks should work", func() {
		testListSize := 10
		testUUID := string(uuid.NewUUID())

		for i := 0; i < testListSize; i++ {
			name := fmt.Sprintf("%s-%d", f.UniqueName, i)
			_, err := createValidatingWebhookConfiguration(f, &admissionregistrationv1.ValidatingWebhookConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name:   name,
					Labels: map[string]string{"e2e-list-test-uuid": testUUID},
				},
				Webhooks: []admissionregistrationv1.ValidatingWebhook{
					newDenyConfigMapWebhookFixture(f, certCtx, servicePort),
					newValidatingIsReadyWebhookFixture(f, certCtx, servicePort),
				},
			})
			framework.ExpectNoError(err, "Creating validating webhook configuration")
		}
		selectorListOpts := metav1.ListOptions{LabelSelector: "e2e-list-test-uuid=" + testUUID}

		ginkgo.By("Listing all of the created validation webhooks")
		list, err := client.AdmissionregistrationV1().ValidatingWebhookConfigurations().List(context.TODO(), selectorListOpts)
		framework.ExpectNoError(err, "Listing validating webhook configurations")
		framework.ExpectEqual(len(list.Items), testListSize)

		// ensure backend is ready before proceeding
		err = waitWebhookConfigurationReady(f)
		framework.ExpectNoError(err, "waiting for webhook configuration to be ready")

		ginkgo.By("Creating a configMap that does not comply to the validation webhook rules")
		err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
			cm := namedNonCompliantConfigMap(string(uuid.NewUUID()), f)
			_, err = client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), cm, metav1.CreateOptions{})
			if err == nil {
				err = client.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), cm.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, "Deleting successfully created configMap")
				return false, nil
			}
			if !strings.Contains(err.Error(), "denied") {
				return false, err
			}
			return true, nil
		})
		framework.ExpectNoError(err, "Waiting for configMap in namespace %s to be denied creation by validating webhook", f.Namespace.Name)

		ginkgo.By("Deleting the collection of validation webhooks")
		err = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, selectorListOpts)
		framework.ExpectNoError(err, "Deleting collection of validating webhook configurations")

		ginkgo.By("Creating a configMap that does not comply to the validation webhook rules")
		err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
			cm := namedNonCompliantConfigMap(string(uuid.NewUUID()), f)
			_, err = client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), cm, metav1.CreateOptions{})
			if err != nil {
				if !strings.Contains(err.Error(), "denied") {
					return false, err
				}
				return false, nil
			}
			err = client.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), cm.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Deleting successfully created configMap")
			return true, nil
		})
		framework.ExpectNoError(err, "Waiting for configMap in namespace %s to be allowed creation since there are no webhooks", f.Namespace.Name)
	})

	/*
		Release: v1.16
		Testname: Admission webhook, list mutating webhooks
		Description: Create 10 mutating webhook configurations, all with a label. Attempt to list the webhook
		configurations matching the label; all the created webhook configurations MUST be present. Attempt to create an
		object; the object MUST be mutated. Attempt to remove the webhook configurations matching the label with deletecollection;
		all webhook configurations MUST be deleted. Attempt to create an object; the object MUST NOT be mutated.
	*/
	framework.ConformanceIt("listing mutating webhooks should work", func() {
		testListSize := 10
		testUUID := string(uuid.NewUUID())

		for i := 0; i < testListSize; i++ {
			name := fmt.Sprintf("%s-%d", f.UniqueName, i)
			_, err := createMutatingWebhookConfiguration(f, &admissionregistrationv1.MutatingWebhookConfiguration{
				ObjectMeta: metav1.ObjectMeta{
					Name:   name,
					Labels: map[string]string{"e2e-list-test-uuid": testUUID},
				},
				Webhooks: []admissionregistrationv1.MutatingWebhook{
					newMutateConfigMapWebhookFixture(f, certCtx, 1, servicePort),
					newMutatingIsReadyWebhookFixture(f, certCtx, servicePort),
				},
			})
			framework.ExpectNoError(err, "Creating mutating webhook configuration")
		}
		selectorListOpts := metav1.ListOptions{LabelSelector: "e2e-list-test-uuid=" + testUUID}

		ginkgo.By("Listing all of the created validation webhooks")
		list, err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().List(context.TODO(), selectorListOpts)
		framework.ExpectNoError(err, "Listing mutating webhook configurations")
		framework.ExpectEqual(len(list.Items), testListSize)

		// ensure backend is ready before proceeding
		err = waitWebhookConfigurationReady(f)
		framework.ExpectNoError(err, "waiting for webhook configuration to be ready")

		ginkgo.By("Creating a configMap that should be mutated")
		err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
			cm := namedToBeMutatedConfigMap(string(uuid.NewUUID()), f)
			created, err := client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), cm, metav1.CreateOptions{})
			if err != nil {
				return false, err
			}
			err = client.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), cm.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Deleting successfully created configMap")
			_, ok := created.Data["mutation-stage-1"]
			return ok, nil
		})
		framework.ExpectNoError(err, "Waiting for configMap in namespace %s to be mutated", f.Namespace.Name)

		ginkgo.By("Deleting the collection of validation webhooks")
		err = client.AdmissionregistrationV1().MutatingWebhookConfigurations().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, selectorListOpts)
		framework.ExpectNoError(err, "Deleting collection of mutating webhook configurations")

		ginkgo.By("Creating a configMap that should not be mutated")
		err = wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
			cm := namedToBeMutatedConfigMap(string(uuid.NewUUID()), f)
			created, err := client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), cm, metav1.CreateOptions{})
			if err != nil {
				return false, err
			}
			err = client.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), cm.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Deleting successfully created configMap")
			_, ok := created.Data["mutation-stage-1"]
			return !ok, nil
		})
		framework.ExpectNoError(err, "Waiting for configMap in namespace %s this is not mutated", f.Namespace.Name)
	})
})

func createAuthReaderRoleBinding(f *framework.Framework, namespace string) {
	ginkgo.By("Create role binding to let webhook read extension-apiserver-authentication")
	client := f.ClientSet
	// Create the role binding to allow the webhook read the extension-apiserver-authentication configmap
	_, err := client.RbacV1().RoleBindings("kube-system").Create(context.TODO(), &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: roleBindingName,
			Annotations: map[string]string{
				rbacv1.AutoUpdateAnnotationKey: "true",
			},
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "",
			Kind:     "Role",
			Name:     "extension-apiserver-authentication-reader",
		},

		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      "default",
				Namespace: namespace,
			},
		},
	}, metav1.CreateOptions{})
	if err != nil && apierrors.IsAlreadyExists(err) {
		framework.Logf("role binding %s already exists", roleBindingName)
	} else {
		framework.ExpectNoError(err, "creating role binding %s:webhook to access configMap", namespace)
	}
}

func deployWebhookAndService(f *framework.Framework, image string, certCtx *certContext, servicePort int32, containerPort int32) {
	ginkgo.By("Deploying the webhook pod")
	client := f.ClientSet

	// Creating the secret that contains the webhook's cert.
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: secretName,
		},
		Type: v1.SecretTypeOpaque,
		Data: map[string][]byte{
			"tls.crt": certCtx.cert,
			"tls.key": certCtx.key,
		},
	}
	namespace := f.Namespace.Name
	_, err := client.CoreV1().Secrets(namespace).Create(context.TODO(), secret, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating secret %q in namespace %q", secretName, namespace)

	// Create the deployment of the webhook
	podLabels := map[string]string{"app": "sample-webhook", "webhook": "true"}
	replicas := int32(1)
	zero := int64(0)
	mounts := []v1.VolumeMount{
		{
			Name:      "webhook-certs",
			ReadOnly:  true,
			MountPath: "/webhook.local.config/certificates",
		},
	}
	volumes := []v1.Volume{
		{
			Name: "webhook-certs",
			VolumeSource: v1.VolumeSource{
				Secret: &v1.SecretVolumeSource{SecretName: secretName},
			},
		},
	}
	containers := []v1.Container{
		{
			Name:         "sample-webhook",
			VolumeMounts: mounts,
			Args: []string{
				"webhook",
				"--tls-cert-file=/webhook.local.config/certificates/tls.crt",
				"--tls-private-key-file=/webhook.local.config/certificates/tls.key",
				"--alsologtostderr",
				"-v=4",
				// Use a non-default port for containers.
				fmt.Sprintf("--port=%d", containerPort),
			},
			ReadinessProbe: &v1.Probe{
				Handler: v1.Handler{
					HTTPGet: &v1.HTTPGetAction{
						Scheme: v1.URISchemeHTTPS,
						Port:   intstr.FromInt(int(containerPort)),
						Path:   "/readyz",
					},
				},
				PeriodSeconds:    1,
				SuccessThreshold: 1,
				FailureThreshold: 30,
			},
			Image: image,
			Ports: []v1.ContainerPort{{ContainerPort: containerPort}},
		},
	}
	d := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   deploymentName,
			Labels: podLabels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: podLabels,
			},
			Strategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabels,
				},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers:                    containers,
					Volumes:                       volumes,
				},
			},
		},
	}
	deployment, err := client.AppsV1().Deployments(namespace).Create(context.TODO(), d, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating deployment %s in namespace %s", deploymentName, namespace)
	ginkgo.By("Wait for the deployment to be ready")
	err = e2edeployment.WaitForDeploymentRevisionAndImage(client, namespace, deploymentName, "1", image)
	framework.ExpectNoError(err, "waiting for the deployment of image %s in %s in %s to complete", image, deploymentName, namespace)
	err = e2edeployment.WaitForDeploymentComplete(client, deployment)
	framework.ExpectNoError(err, "waiting for the deployment status valid", image, deploymentName, namespace)

	ginkgo.By("Deploying the webhook service")

	serviceLabels := map[string]string{"webhook": "true"}
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      serviceName,
			Labels:    map[string]string{"test": "webhook"},
		},
		Spec: v1.ServiceSpec{
			Selector: serviceLabels,
			Ports: []v1.ServicePort{
				{
					Protocol:   "TCP",
					Port:       servicePort,
					TargetPort: intstr.FromInt(int(containerPort)),
				},
			},
		},
	}
	_, err = client.CoreV1().Services(namespace).Create(context.TODO(), service, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating service %s in namespace %s", serviceName, namespace)

	ginkgo.By("Verifying the service has paired with the endpoint")
	err = framework.WaitForServiceEndpointsNum(client, namespace, serviceName, 1, 1*time.Second, 30*time.Second)
	framework.ExpectNoError(err, "waiting for service %s/%s have %d endpoint", namespace, serviceName, 1)
}

func strPtr(s string) *string { return &s }

func registerWebhook(f *framework.Framework, configName string, certCtx *certContext, servicePort int32) func() {
	client := f.ClientSet
	ginkgo.By("Registering the webhook via the AdmissionRegistration API")

	namespace := f.Namespace.Name
	// A webhook that cannot talk to server, with fail-open policy
	failOpenHook := failingWebhook(namespace, "fail-open.k8s.io", servicePort)
	policyIgnore := admissionregistrationv1.Ignore
	failOpenHook.FailurePolicy = &policyIgnore
	failOpenHook.NamespaceSelector = &metav1.LabelSelector{
		MatchLabels: map[string]string{f.UniqueName: "true"},
	}

	_, err := createValidatingWebhookConfiguration(f, &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			newDenyPodWebhookFixture(f, certCtx, servicePort),
			newDenyConfigMapWebhookFixture(f, certCtx, servicePort),
			// Server cannot talk to this webhook, so it always fails.
			// Because this webhook is configured fail-open, request should be admitted after the call fails.
			failOpenHook,

			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newValidatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering webhook config %s with namespace %s", configName, namespace)

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")

	return func() {
		client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
	}
}

func registerWebhookForAttachingPod(f *framework.Framework, configName string, certCtx *certContext, servicePort int32) func() {
	client := f.ClientSet
	ginkgo.By("Registering the webhook via the AdmissionRegistration API")

	namespace := f.Namespace.Name
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone

	_, err := createValidatingWebhookConfiguration(f, &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name: "deny-attaching-pod.k8s.io",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Connect},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{""},
						APIVersions: []string{"v1"},
						Resources:   []string{"pods/attach"},
					},
				}},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: namespace,
						Name:      serviceName,
						Path:      strPtr("/pods/attach"),
						Port:      pointer.Int32Ptr(servicePort),
					},
					CABundle: certCtx.signingCert,
				},
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1", "v1beta1"},
				// Scope the webhook to just this namespace
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{f.UniqueName: "true"},
				},
			},
			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newValidatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering webhook config %s with namespace %s", configName, namespace)

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")

	return func() {
		client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
	}
}

func registerMutatingWebhookForConfigMap(f *framework.Framework, configName string, certCtx *certContext, servicePort int32) func() {
	client := f.ClientSet
	ginkgo.By("Registering the mutating configmap webhook via the AdmissionRegistration API")

	namespace := f.Namespace.Name

	_, err := createMutatingWebhookConfiguration(f, &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.MutatingWebhook{
			newMutateConfigMapWebhookFixture(f, certCtx, 1, servicePort),
			newMutateConfigMapWebhookFixture(f, certCtx, 2, servicePort),
			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newMutatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering mutating webhook config %s with namespace %s", configName, namespace)

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")
	return func() {
		client.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
	}
}

func testMutatingConfigMapWebhook(f *framework.Framework) {
	ginkgo.By("create a configmap that should be updated by the webhook")
	client := f.ClientSet
	configMap := toBeMutatedConfigMap(f)
	mutatedConfigMap, err := client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	expectedConfigMapData := map[string]string{
		"mutation-start":   "yes",
		"mutation-stage-1": "yes",
		"mutation-stage-2": "yes",
	}
	if !reflect.DeepEqual(expectedConfigMapData, mutatedConfigMap.Data) {
		framework.Failf("\nexpected %#v\n, got %#v\n", expectedConfigMapData, mutatedConfigMap.Data)
	}
}

func registerMutatingWebhookForPod(f *framework.Framework, configName string, certCtx *certContext, servicePort int32) func() {
	client := f.ClientSet
	ginkgo.By("Registering the mutating pod webhook via the AdmissionRegistration API")

	namespace := f.Namespace.Name
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone

	_, err := createMutatingWebhookConfiguration(f, &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.MutatingWebhook{
			{
				Name: "adding-init-container.k8s.io",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{""},
						APIVersions: []string{"v1"},
						Resources:   []string{"pods"},
					},
				}},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: namespace,
						Name:      serviceName,
						Path:      strPtr("/mutating-pods"),
						Port:      pointer.Int32Ptr(servicePort),
					},
					CABundle: certCtx.signingCert,
				},
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1", "v1beta1"},
				// Scope the webhook to just this namespace
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{f.UniqueName: "true"},
				},
			},
			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newMutatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering mutating webhook config %s with namespace %s", configName, namespace)

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")

	return func() {
		client.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
	}
}

func testMutatingPodWebhook(f *framework.Framework) {
	ginkgo.By("create a pod that should be updated by the webhook")
	client := f.ClientSet
	pod := toBeMutatedPod(f)
	mutatedPod, err := client.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	gomega.Expect(err).To(gomega.BeNil())
	if len(mutatedPod.Spec.InitContainers) != 1 {
		framework.Failf("expect pod to have 1 init container, got %#v", mutatedPod.Spec.InitContainers)
	}
	if got, expected := mutatedPod.Spec.InitContainers[0].Name, "webhook-added-init-container"; got != expected {
		framework.Failf("expect the init container name to be %q, got %q", expected, got)
	}
	if got, expected := mutatedPod.Spec.InitContainers[0].TerminationMessagePolicy, v1.TerminationMessageReadFile; got != expected {
		framework.Failf("expect the init terminationMessagePolicy to be default to %q, got %q", expected, got)
	}
}

func toBeMutatedPod(f *framework.Framework) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "webhook-to-be-mutated",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "example",
					Image: imageutils.GetPauseImageName(),
				},
			},
		},
	}
}

func testWebhook(f *framework.Framework) {
	ginkgo.By("create a pod that should be denied by the webhook")
	client := f.ClientSet
	// Creating the pod, the request should be rejected
	pod := nonCompliantPod(f)
	_, err := client.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectError(err, "create pod %s in namespace %s should have been denied by webhook", pod.Name, f.Namespace.Name)
	expectedErrMsg1 := "the pod contains unwanted container name"
	if !strings.Contains(err.Error(), expectedErrMsg1) {
		framework.Failf("expect error contains %q, got %q", expectedErrMsg1, err.Error())
	}
	expectedErrMsg2 := "the pod contains unwanted label"
	if !strings.Contains(err.Error(), expectedErrMsg2) {
		framework.Failf("expect error contains %q, got %q", expectedErrMsg2, err.Error())
	}

	ginkgo.By("create a pod that causes the webhook to hang")
	client = f.ClientSet
	// Creating the pod, the request should be rejected
	pod = hangingPod(f)
	_, err = client.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectError(err, "create pod %s in namespace %s should have caused webhook to hang", pod.Name, f.Namespace.Name)
	// ensure the error is webhook-related, not client-side
	if !strings.Contains(err.Error(), "webhook") {
		framework.Failf("expect error %q, got %q", "webhook", err.Error())
	}
	// ensure the error is a timeout
	if !strings.Contains(err.Error(), "deadline") {
		framework.Failf("expect error %q, got %q", "deadline", err.Error())
	}
	// ensure the pod was not actually created
	if _, err := client.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), pod.Name, metav1.GetOptions{}); !apierrors.IsNotFound(err) {
		framework.Failf("expect notfound error looking for rejected pod, got %v", err)
	}

	ginkgo.By("create a configmap that should be denied by the webhook")
	// Creating the configmap, the request should be rejected
	configmap := nonCompliantConfigMap(f)
	_, err = client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configmap, metav1.CreateOptions{})
	framework.ExpectError(err, "create configmap %s in namespace %s should have been denied by the webhook", configmap.Name, f.Namespace.Name)
	expectedErrMsg := "the configmap contains unwanted key and value"
	if !strings.Contains(err.Error(), expectedErrMsg) {
		framework.Failf("expect error contains %q, got %q", expectedErrMsg, err.Error())
	}

	ginkgo.By("create a configmap that should be admitted by the webhook")
	// Creating the configmap, the request should be admitted
	configmap = &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: allowedConfigMapName,
		},
		Data: map[string]string{
			"admit": "this",
		},
	}
	_, err = client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configmap, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create configmap %s in namespace: %s", configmap.Name, f.Namespace.Name)

	ginkgo.By("update (PUT) the admitted configmap to a non-compliant one should be rejected by the webhook")
	toNonCompliantFn := func(cm *v1.ConfigMap) {
		if cm.Data == nil {
			cm.Data = map[string]string{}
		}
		cm.Data["webhook-e2e-test"] = "webhook-disallow"
	}
	_, err = updateConfigMap(client, f.Namespace.Name, allowedConfigMapName, toNonCompliantFn)
	framework.ExpectError(err, "update (PUT) admitted configmap %s in namespace %s to a non-compliant one should be rejected by webhook", allowedConfigMapName, f.Namespace.Name)
	if !strings.Contains(err.Error(), expectedErrMsg) {
		framework.Failf("expect error contains %q, got %q", expectedErrMsg, err.Error())
	}

	ginkgo.By("update (PATCH) the admitted configmap to a non-compliant one should be rejected by the webhook")
	patch := nonCompliantConfigMapPatch()
	_, err = client.CoreV1().ConfigMaps(f.Namespace.Name).Patch(context.TODO(), allowedConfigMapName, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
	framework.ExpectError(err, "update admitted configmap %s in namespace %s by strategic merge patch to a non-compliant one should be rejected by webhook. Patch: %+v", allowedConfigMapName, f.Namespace.Name, patch)
	if !strings.Contains(err.Error(), expectedErrMsg) {
		framework.Failf("expect error contains %q, got %q", expectedErrMsg, err.Error())
	}

	ginkgo.By("create a namespace that bypass the webhook")
	err = createNamespace(f, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{
		Name: skippedNamespaceName,
		Labels: map[string]string{
			skipNamespaceLabelKey: skipNamespaceLabelValue,
			f.UniqueName:          "true",
		},
	}})
	framework.ExpectNoError(err, "creating namespace %q", skippedNamespaceName)
	// clean up the namespace
	defer client.CoreV1().Namespaces().Delete(context.TODO(), skippedNamespaceName, metav1.DeleteOptions{})

	ginkgo.By("create a configmap that violates the webhook policy but is in a whitelisted namespace")
	configmap = nonCompliantConfigMap(f)
	_, err = client.CoreV1().ConfigMaps(skippedNamespaceName).Create(context.TODO(), configmap, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create configmap %s in namespace: %s", configmap.Name, skippedNamespaceName)
}

func testAttachingPodWebhook(f *framework.Framework) {
	ginkgo.By("create a pod")
	client := f.ClientSet
	pod := toBeAttachedPod(f)
	_, err := client.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create pod %s in namespace: %s", pod.Name, f.Namespace.Name)
	err = e2epod.WaitForPodNameRunningInNamespace(client, pod.Name, f.Namespace.Name)
	framework.ExpectNoError(err, "error while waiting for pod %s to go to Running phase in namespace: %s", pod.Name, f.Namespace.Name)

	ginkgo.By("'kubectl attach' the pod, should be denied by the webhook")
	timer := time.NewTimer(30 * time.Second)
	defer timer.Stop()
	_, err = framework.NewKubectlCommand(f.Namespace.Name, "attach", fmt.Sprintf("--namespace=%v", f.Namespace.Name), pod.Name, "-i", "-c=container1").WithTimeout(timer.C).Exec()
	framework.ExpectError(err, "'kubectl attach' the pod, should be denied by the webhook")
	if e, a := "attaching to pod 'to-be-attached-pod' is not allowed", err.Error(); !strings.Contains(a, e) {
		framework.Failf("unexpected 'kubectl attach' error message. expected to contain %q, got %q", e, a)
	}
}

// failingWebhook returns a webhook with rule of create configmaps,
// but with an invalid client config so that server cannot communicate with it
func failingWebhook(namespace, name string, servicePort int32) admissionregistrationv1.ValidatingWebhook {
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone

	return admissionregistrationv1.ValidatingWebhook{
		Name: name,
		Rules: []admissionregistrationv1.RuleWithOperations{{
			Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
			Rule: admissionregistrationv1.Rule{
				APIGroups:   []string{""},
				APIVersions: []string{"v1"},
				Resources:   []string{"configmaps"},
			},
		}},
		ClientConfig: admissionregistrationv1.WebhookClientConfig{
			Service: &admissionregistrationv1.ServiceReference{
				Namespace: namespace,
				Name:      serviceName,
				Path:      strPtr("/configmaps"),
				Port:      pointer.Int32Ptr(servicePort),
			},
			// Without CA bundle, the call to webhook always fails
			CABundle: nil,
		},
		SideEffects:             &sideEffectsNone,
		AdmissionReviewVersions: []string{"v1", "v1beta1"},
	}
}

func registerFailClosedWebhook(f *framework.Framework, configName string, certCtx *certContext, servicePort int32) func() {
	ginkgo.By("Registering a webhook that server cannot talk to, with fail closed policy, via the AdmissionRegistration API")

	namespace := f.Namespace.Name
	// A webhook that cannot talk to server, with fail-closed policy
	policyFail := admissionregistrationv1.Fail
	hook := failingWebhook(namespace, "fail-closed.k8s.io", servicePort)
	hook.FailurePolicy = &policyFail
	hook.NamespaceSelector = &metav1.LabelSelector{
		MatchLabels: map[string]string{f.UniqueName: "true"},
		MatchExpressions: []metav1.LabelSelectorRequirement{
			{
				Key:      failNamespaceLabelKey,
				Operator: metav1.LabelSelectorOpIn,
				Values:   []string{failNamespaceLabelValue},
			},
		},
	}

	_, err := createValidatingWebhookConfiguration(f, &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			// Server cannot talk to this webhook, so it always fails.
			// Because this webhook is configured fail-closed, request should be rejected after the call fails.
			hook,
			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newValidatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering webhook config %s with namespace %s", configName, namespace)

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")
	return func() {
		f.ClientSet.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
	}
}

func testFailClosedWebhook(f *framework.Framework) {
	client := f.ClientSet
	ginkgo.By("create a namespace for the webhook")
	err := createNamespace(f, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{
		Name: failNamespaceName,
		Labels: map[string]string{
			failNamespaceLabelKey: failNamespaceLabelValue,
			f.UniqueName:          "true",
		},
	}})
	framework.ExpectNoError(err, "creating namespace %q", failNamespaceName)
	defer client.CoreV1().Namespaces().Delete(context.TODO(), failNamespaceName, metav1.DeleteOptions{})

	ginkgo.By("create a configmap should be unconditionally rejected by the webhook")
	configmap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
	}
	_, err = client.CoreV1().ConfigMaps(failNamespaceName).Create(context.TODO(), configmap, metav1.CreateOptions{})
	framework.ExpectError(err, "create configmap in namespace: %s should be unconditionally rejected by the webhook", failNamespaceName)
	if !apierrors.IsInternalError(err) {
		framework.Failf("expect an internal error, got %#v", err)
	}
}

func registerValidatingWebhookForWebhookConfigurations(f *framework.Framework, configName string, certCtx *certContext, servicePort int32) func() {
	var err error
	client := f.ClientSet
	ginkgo.By("Registering a validating webhook on ValidatingWebhookConfiguration and MutatingWebhookConfiguration objects, via the AdmissionRegistration API")

	namespace := f.Namespace.Name
	failurePolicy := admissionregistrationv1.Fail
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone

	// This webhook denies all requests to Delete validating webhook configuration and
	// mutating webhook configuration objects. It should never be called, however, because
	// dynamic admission webhooks should not be called on requests involving webhook configuration objects.
	_, err = createValidatingWebhookConfiguration(f, &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name: "deny-webhook-configuration-deletions.k8s.io",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Delete},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{"admissionregistration.k8s.io"},
						APIVersions: []string{"*"},
						Resources: []string{
							"validatingwebhookconfigurations",
							"mutatingwebhookconfigurations",
						},
					},
				}},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: namespace,
						Name:      serviceName,
						Path:      strPtr("/always-deny"),
						Port:      pointer.Int32Ptr(servicePort),
					},
					CABundle: certCtx.signingCert,
				},
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1", "v1beta1"},
				FailurePolicy:           &failurePolicy,
				// Scope the webhook to just this namespace
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{f.UniqueName: "true"},
				},
			},
			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newValidatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering webhook config %s with namespace %s", configName, namespace)

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")
	return func() {
		err := client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "deleting webhook config %s with namespace %s", configName, namespace)
	}
}

func registerMutatingWebhookForWebhookConfigurations(f *framework.Framework, configName string, certCtx *certContext, servicePort int32) func() {
	var err error
	client := f.ClientSet
	ginkgo.By("Registering a mutating webhook on ValidatingWebhookConfiguration and MutatingWebhookConfiguration objects, via the AdmissionRegistration API")

	namespace := f.Namespace.Name
	failurePolicy := admissionregistrationv1.Fail
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone

	// This webhook adds a label to all requests create to validating webhook configuration and
	// mutating webhook configuration objects. It should never be called, however, because
	// dynamic admission webhooks should not be called on requests involving webhook configuration objects.
	_, err = createMutatingWebhookConfiguration(f, &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.MutatingWebhook{
			{
				Name: "add-label-to-webhook-configurations.k8s.io",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{"admissionregistration.k8s.io"},
						APIVersions: []string{"*"},
						Resources: []string{
							"validatingwebhookconfigurations",
							"mutatingwebhookconfigurations",
						},
					},
				}},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: namespace,
						Name:      serviceName,
						Path:      strPtr("/add-label"),
						Port:      pointer.Int32Ptr(servicePort),
					},
					CABundle: certCtx.signingCert,
				},
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1", "v1beta1"},
				FailurePolicy:           &failurePolicy,
				// Scope the webhook to just this namespace
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{f.UniqueName: "true"},
				},
			},
			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newMutatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering webhook config %s with namespace %s", configName, namespace)

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")
	return func() {
		err := client.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "deleting webhook config %s with namespace %s", configName, namespace)
	}
}

// This test assumes that the deletion-rejecting webhook defined in
// registerValidatingWebhookForWebhookConfigurations and the webhook-config-mutating
// webhook defined in registerMutatingWebhookForWebhookConfigurations already exist.
func testWebhooksForWebhookConfigurations(f *framework.Framework, configName string, certCtx *certContext, servicePort int32) {
	var err error
	client := f.ClientSet
	ginkgo.By("Creating a dummy validating-webhook-configuration object")

	namespace := f.Namespace.Name
	failurePolicy := admissionregistrationv1.Ignore
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone

	mutatedValidatingWebhookConfiguration, err := createValidatingWebhookConfiguration(f, &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name: "dummy-validating-webhook.k8s.io",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					// This will not match any real resources so this webhook should never be called.
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{""},
						APIVersions: []string{"v1"},
						Resources:   []string{"invalid"},
					},
				}},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: namespace,
						Name:      serviceName,
						// This path not recognized by the webhook service,
						// so the call to this webhook will always fail,
						// but because the failure policy is ignore, it will
						// have no effect on admission requests.
						Path: strPtr(""),
						Port: pointer.Int32Ptr(servicePort),
					},
					CABundle: nil,
				},
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1", "v1beta1"},
				FailurePolicy:           &failurePolicy,
				// Scope the webhook to just this namespace
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{f.UniqueName: "true"},
				},
			},
			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newValidatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering webhook config %s with namespace %s", configName, namespace)
	if mutatedValidatingWebhookConfiguration.ObjectMeta.Labels != nil && mutatedValidatingWebhookConfiguration.ObjectMeta.Labels[addedLabelKey] == addedLabelValue {
		framework.Failf("expected %s not to be mutated by mutating webhooks but it was", configName)
	}

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")

	ginkgo.By("Deleting the validating-webhook-configuration, which should be possible to remove")

	err = client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
	framework.ExpectNoError(err, "deleting webhook config %s with namespace %s", configName, namespace)

	ginkgo.By("Creating a dummy mutating-webhook-configuration object")

	mutatedMutatingWebhookConfiguration, err := createMutatingWebhookConfiguration(f, &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.MutatingWebhook{
			{
				Name: "dummy-mutating-webhook.k8s.io",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					// This will not match any real resources so this webhook should never be called.
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{""},
						APIVersions: []string{"v1"},
						Resources:   []string{"invalid"},
					},
				}},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: namespace,
						Name:      serviceName,
						// This path not recognized by the webhook service,
						// so the call to this webhook will always fail,
						// but because the failure policy is ignore, it will
						// have no effect on admission requests.
						Path: strPtr(""),
						Port: pointer.Int32Ptr(servicePort),
					},
					CABundle: nil,
				},
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1", "v1beta1"},
				FailurePolicy:           &failurePolicy,
				// Scope the webhook to just this namespace
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{f.UniqueName: "true"},
				},
			},
			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newMutatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering webhook config %s with namespace %s", configName, namespace)
	if mutatedMutatingWebhookConfiguration.ObjectMeta.Labels != nil && mutatedMutatingWebhookConfiguration.ObjectMeta.Labels[addedLabelKey] == addedLabelValue {
		framework.Failf("expected %s not to be mutated by mutating webhooks but it was", configName)
	}

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")

	ginkgo.By("Deleting the mutating-webhook-configuration, which should be possible to remove")

	err = client.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
	framework.ExpectNoError(err, "deleting webhook config %s with namespace %s", configName, namespace)
}

func createNamespace(f *framework.Framework, ns *v1.Namespace) error {
	return wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		_, err := f.ClientSet.CoreV1().Namespaces().Create(context.TODO(), ns, metav1.CreateOptions{})
		if err != nil {
			if strings.HasPrefix(err.Error(), "object is being deleted:") {
				return false, nil
			}
			return false, err
		}
		return true, nil
	})
}

func nonCompliantPod(f *framework.Framework) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: disallowedPodName,
			Labels: map[string]string{
				"webhook-e2e-test": "webhook-disallow",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "webhook-disallow",
					Image: imageutils.GetPauseImageName(),
				},
			},
		},
	}
}

func hangingPod(f *framework.Framework) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: hangingPodName,
			Labels: map[string]string{
				"webhook-e2e-test": "wait-forever",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "wait-forever",
					Image: imageutils.GetPauseImageName(),
				},
			},
		},
	}
}

func toBeAttachedPod(f *framework.Framework) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: toBeAttachedPodName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "container1",
					Image: imageutils.GetPauseImageName(),
				},
			},
		},
	}
}

func nonCompliantConfigMap(f *framework.Framework) *v1.ConfigMap {
	return namedNonCompliantConfigMap(disallowedConfigMapName, f)
}

func namedNonCompliantConfigMap(name string, f *framework.Framework) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Data: map[string]string{
			"webhook-e2e-test": "webhook-disallow",
		},
	}
}

func toBeMutatedConfigMap(f *framework.Framework) *v1.ConfigMap {
	return namedToBeMutatedConfigMap("to-be-mutated", f)
}

func namedToBeMutatedConfigMap(name string, f *framework.Framework) *v1.ConfigMap {
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Data: map[string]string{
			"mutation-start": "yes",
		},
	}
}

func nonCompliantConfigMapPatch() string {
	return fmt.Sprint(`{"data":{"webhook-e2e-test":"webhook-disallow"}}`)
}

type updateConfigMapFn func(cm *v1.ConfigMap)

func updateConfigMap(c clientset.Interface, ns, name string, update updateConfigMapFn) (*v1.ConfigMap, error) {
	var cm *v1.ConfigMap
	pollErr := wait.PollImmediate(2*time.Second, 1*time.Minute, func() (bool, error) {
		var err error
		if cm, err = c.CoreV1().ConfigMaps(ns).Get(context.TODO(), name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		update(cm)
		if cm, err = c.CoreV1().ConfigMaps(ns).Update(context.TODO(), cm, metav1.UpdateOptions{}); err == nil {
			return true, nil
		}
		// Only retry update on conflict
		if !apierrors.IsConflict(err) {
			return false, err
		}
		return false, nil
	})
	return cm, pollErr
}

type updateCustomResourceFn func(cm *unstructured.Unstructured)

func updateCustomResource(c dynamic.ResourceInterface, ns, name string, update updateCustomResourceFn) (*unstructured.Unstructured, error) {
	var cr *unstructured.Unstructured
	pollErr := wait.PollImmediate(2*time.Second, 1*time.Minute, func() (bool, error) {
		var err error
		if cr, err = c.Get(context.TODO(), name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		update(cr)
		if cr, err = c.Update(context.TODO(), cr, metav1.UpdateOptions{}); err == nil {
			return true, nil
		}
		// Only retry update on conflict
		if !apierrors.IsConflict(err) {
			return false, err
		}
		return false, nil
	})
	return cr, pollErr
}

func cleanWebhookTest(client clientset.Interface, namespaceName string) {
	_ = client.CoreV1().Services(namespaceName).Delete(context.TODO(), serviceName, metav1.DeleteOptions{})
	_ = client.AppsV1().Deployments(namespaceName).Delete(context.TODO(), deploymentName, metav1.DeleteOptions{})
	_ = client.CoreV1().Secrets(namespaceName).Delete(context.TODO(), secretName, metav1.DeleteOptions{})
	_ = client.RbacV1().RoleBindings("kube-system").Delete(context.TODO(), roleBindingName, metav1.DeleteOptions{})
}

func registerWebhookForCustomResource(f *framework.Framework, configName string, certCtx *certContext, testcrd *crd.TestCrd, servicePort int32) func() {
	client := f.ClientSet
	ginkgo.By("Registering the custom resource webhook via the AdmissionRegistration API")

	namespace := f.Namespace.Name
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone

	_, err := createValidatingWebhookConfiguration(f, &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name: "deny-unwanted-custom-resource-data.k8s.io",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update, admissionregistrationv1.Delete},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{testcrd.Crd.Spec.Group},
						APIVersions: servedAPIVersions(testcrd.Crd),
						Resources:   []string{testcrd.Crd.Spec.Names.Plural},
					},
				}},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: namespace,
						Name:      serviceName,
						Path:      strPtr("/custom-resource"),
						Port:      pointer.Int32Ptr(servicePort),
					},
					CABundle: certCtx.signingCert,
				},
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1", "v1beta1"},
				// Scope the webhook to just this namespace
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{f.UniqueName: "true"},
				},
			},
			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newValidatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering custom resource webhook config %s with namespace %s", configName, namespace)

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")
	return func() {
		client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
	}
}

func registerMutatingWebhookForCustomResource(f *framework.Framework, configName string, certCtx *certContext, testcrd *crd.TestCrd, servicePort int32) func() {
	client := f.ClientSet
	ginkgo.By(fmt.Sprintf("Registering the mutating webhook for custom resource %s via the AdmissionRegistration API", testcrd.Crd.Name))

	namespace := f.Namespace.Name
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone

	_, err := createMutatingWebhookConfiguration(f, &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.MutatingWebhook{
			{
				Name: "mutate-custom-resource-data-stage-1.k8s.io",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{testcrd.Crd.Spec.Group},
						APIVersions: servedAPIVersions(testcrd.Crd),
						Resources:   []string{testcrd.Crd.Spec.Names.Plural},
					},
				}},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: namespace,
						Name:      serviceName,
						Path:      strPtr("/mutating-custom-resource"),
						Port:      pointer.Int32Ptr(servicePort),
					},
					CABundle: certCtx.signingCert,
				},
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1", "v1beta1"},
				// Scope the webhook to just this namespace
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{f.UniqueName: "true"},
				},
			},
			{
				Name: "mutate-custom-resource-data-stage-2.k8s.io",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{testcrd.Crd.Spec.Group},
						APIVersions: servedAPIVersions(testcrd.Crd),
						Resources:   []string{testcrd.Crd.Spec.Names.Plural},
					},
				}},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: namespace,
						Name:      serviceName,
						Path:      strPtr("/mutating-custom-resource"),
						Port:      pointer.Int32Ptr(servicePort),
					},
					CABundle: certCtx.signingCert,
				},
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1", "v1beta1"},
				// Scope the webhook to just this namespace
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{f.UniqueName: "true"},
				},
			},
			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newMutatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering custom resource webhook config %s with namespace %s", configName, namespace)

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")

	return func() {
		client.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
	}
}

func testCustomResourceWebhook(f *framework.Framework, crd *apiextensionsv1.CustomResourceDefinition, customResourceClient dynamic.ResourceInterface) {
	ginkgo.By("Creating a custom resource that should be denied by the webhook")
	crInstanceName := "cr-instance-1"
	crInstance := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       crd.Spec.Names.Kind,
			"apiVersion": crd.Spec.Group + "/" + crd.Spec.Versions[0].Name,
			"metadata": map[string]interface{}{
				"name":      crInstanceName,
				"namespace": f.Namespace.Name,
			},
			"data": map[string]interface{}{
				"webhook-e2e-test": "webhook-disallow",
			},
		},
	}
	_, err := customResourceClient.Create(context.TODO(), crInstance, metav1.CreateOptions{})
	framework.ExpectError(err, "create custom resource %s in namespace %s should be denied by webhook", crInstanceName, f.Namespace.Name)
	expectedErrMsg := "the custom resource contains unwanted data"
	if !strings.Contains(err.Error(), expectedErrMsg) {
		framework.Failf("expect error contains %q, got %q", expectedErrMsg, err.Error())
	}
}

func testBlockingCustomResourceUpdateDeletion(f *framework.Framework, crd *apiextensionsv1.CustomResourceDefinition, customResourceClient dynamic.ResourceInterface) {
	ginkgo.By("Creating a custom resource whose deletion would be denied by the webhook")
	crInstanceName := "cr-instance-2"
	crInstance := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       crd.Spec.Names.Kind,
			"apiVersion": crd.Spec.Group + "/" + crd.Spec.Versions[0].Name,
			"metadata": map[string]interface{}{
				"name":      crInstanceName,
				"namespace": f.Namespace.Name,
			},
			"data": map[string]interface{}{
				"webhook-e2e-test": "webhook-nondeletable",
			},
		},
	}
	_, err := customResourceClient.Create(context.TODO(), crInstance, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create custom resource %s in namespace: %s", crInstanceName, f.Namespace.Name)

	ginkgo.By("Updating the custom resource with disallowed data should be denied")
	toNonCompliantFn := func(cr *unstructured.Unstructured) {
		if _, ok := cr.Object["data"]; !ok {
			cr.Object["data"] = map[string]interface{}{}
		}
		data := cr.Object["data"].(map[string]interface{})
		data["webhook-e2e-test"] = "webhook-disallow"
	}
	_, err = updateCustomResource(customResourceClient, f.Namespace.Name, crInstanceName, toNonCompliantFn)
	framework.ExpectError(err, "updating custom resource %s in namespace: %s should be denied", crInstanceName, f.Namespace.Name)

	expectedErrMsg := "the custom resource contains unwanted data"
	if !strings.Contains(err.Error(), expectedErrMsg) {
		framework.Failf("expect error contains %q, got %q", expectedErrMsg, err.Error())
	}

	ginkgo.By("Deleting the custom resource should be denied")
	err = customResourceClient.Delete(context.TODO(), crInstanceName, metav1.DeleteOptions{})
	framework.ExpectError(err, "deleting custom resource %s in namespace: %s should be denied", crInstanceName, f.Namespace.Name)
	expectedErrMsg1 := "the custom resource cannot be deleted because it contains unwanted key and value"
	if !strings.Contains(err.Error(), expectedErrMsg1) {
		framework.Failf("expect error contains %q, got %q", expectedErrMsg1, err.Error())
	}

	ginkgo.By("Remove the offending key and value from the custom resource data")
	toCompliantFn := func(cr *unstructured.Unstructured) {
		if _, ok := cr.Object["data"]; !ok {
			cr.Object["data"] = map[string]interface{}{}
		}
		data := cr.Object["data"].(map[string]interface{})
		data["webhook-e2e-test"] = "webhook-allow"
	}
	_, err = updateCustomResource(customResourceClient, f.Namespace.Name, crInstanceName, toCompliantFn)
	framework.ExpectNoError(err, "failed to update custom resource %s in namespace: %s", crInstanceName, f.Namespace.Name)

	ginkgo.By("Deleting the updated custom resource should be successful")
	err = customResourceClient.Delete(context.TODO(), crInstanceName, metav1.DeleteOptions{})
	framework.ExpectNoError(err, "failed to delete custom resource %s in namespace: %s", crInstanceName, f.Namespace.Name)

}

func testMutatingCustomResourceWebhook(f *framework.Framework, crd *apiextensionsv1.CustomResourceDefinition, customResourceClient dynamic.ResourceInterface, prune bool) {
	ginkgo.By("Creating a custom resource that should be mutated by the webhook")
	crName := "cr-instance-1"
	cr := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       crd.Spec.Names.Kind,
			"apiVersion": crd.Spec.Group + "/" + crd.Spec.Versions[0].Name,
			"metadata": map[string]interface{}{
				"name":      crName,
				"namespace": f.Namespace.Name,
			},
			"data": map[string]interface{}{
				"mutation-start": "yes",
			},
		},
	}
	mutatedCR, err := customResourceClient.Create(context.TODO(), cr, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create custom resource %s in namespace: %s", crName, f.Namespace.Name)
	expectedCRData := map[string]interface{}{
		"mutation-start":   "yes",
		"mutation-stage-1": "yes",
	}
	if !prune {
		expectedCRData["mutation-stage-2"] = "yes"
	}
	if !reflect.DeepEqual(expectedCRData, mutatedCR.Object["data"]) {
		framework.Failf("\nexpected %#v\n, got %#v\n", expectedCRData, mutatedCR.Object["data"])
	}
}

func testMultiVersionCustomResourceWebhook(f *framework.Framework, testcrd *crd.TestCrd) {
	customResourceClient := testcrd.DynamicClients["v1"]
	ginkgo.By("Creating a custom resource while v1 is storage version")
	crName := "cr-instance-1"
	cr := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       testcrd.Crd.Spec.Names.Kind,
			"apiVersion": testcrd.Crd.Spec.Group + "/" + testcrd.Crd.Spec.Versions[0].Name,
			"metadata": map[string]interface{}{
				"name":      crName,
				"namespace": f.Namespace.Name,
			},
			"data": map[string]interface{}{
				"mutation-start": "yes",
			},
		},
	}
	_, err := customResourceClient.Create(context.TODO(), cr, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create custom resource %s in namespace: %s", crName, f.Namespace.Name)

	ginkgo.By("Patching Custom Resource Definition to set v2 as storage")
	apiVersionWithV2StoragePatch := `{
		"spec": {
		  "versions": [
		    {
			  "name": "v1",
			  "storage": false,
			  "served": true,
			  "schema": {
			    "openAPIV3Schema": {"x-kubernetes-preserve-unknown-fields": true, "type": "object"}
			  }
            },
		    {
			  "name": "v2",
			  "storage": true,
			  "served": true,
			  "schema": {
			    "openAPIV3Schema": {"x-kubernetes-preserve-unknown-fields": true, "type": "object"}
			  }
            }
          ]
       }
    }`
	_, err = testcrd.APIExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Patch(context.TODO(), testcrd.Crd.Name, types.StrategicMergePatchType, []byte(apiVersionWithV2StoragePatch), metav1.PatchOptions{})
	framework.ExpectNoError(err, "failed to patch custom resource definition %s in namespace: %s", testcrd.Crd.Name, f.Namespace.Name)

	ginkgo.By("Patching the custom resource while v2 is storage version")
	crDummyPatch := fmt.Sprint(`[{ "op": "add", "path": "/dummy", "value": "test" }]`)
	mutatedCR, err := testcrd.DynamicClients["v2"].Patch(context.TODO(), crName, types.JSONPatchType, []byte(crDummyPatch), metav1.PatchOptions{})
	framework.ExpectNoError(err, "failed to patch custom resource %s in namespace: %s", crName, f.Namespace.Name)
	expectedCRData := map[string]interface{}{
		"mutation-start":   "yes",
		"mutation-stage-1": "yes",
		"mutation-stage-2": "yes",
	}
	if !reflect.DeepEqual(expectedCRData, mutatedCR.Object["data"]) {
		framework.Failf("\nexpected %#v\n, got %#v\n", expectedCRData, mutatedCR.Object["data"])
	}
	if !reflect.DeepEqual("test", mutatedCR.Object["dummy"]) {
		framework.Failf("\nexpected %#v\n, got %#v\n", "test", mutatedCR.Object["dummy"])
	}
}

func registerValidatingWebhookForCRD(f *framework.Framework, configName string, certCtx *certContext, servicePort int32) func() {
	client := f.ClientSet
	ginkgo.By("Registering the crd webhook via the AdmissionRegistration API")

	namespace := f.Namespace.Name
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone

	// This webhook will deny the creation of CustomResourceDefinitions which have the
	// label "webhook-e2e-test":"webhook-disallow"
	// NOTE: Because tests are run in parallel and in an unpredictable order, it is critical
	// that no other test attempts to create CRD with that label.
	_, err := createValidatingWebhookConfiguration(f, &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name: "deny-crd-with-unwanted-label.k8s.io",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{"apiextensions.k8s.io"},
						APIVersions: []string{"*"},
						Resources:   []string{"customresourcedefinitions"},
					},
				}},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: namespace,
						Name:      serviceName,
						Path:      strPtr("/crd"),
						Port:      pointer.Int32Ptr(servicePort),
					},
					CABundle: certCtx.signingCert,
				},
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1", "v1beta1"},
				// Scope the webhook to just this test
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{f.UniqueName: "true"},
				},
			},
			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newValidatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering crd webhook config %s with namespace %s", configName, namespace)

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")
	return func() {
		client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
	}
}

func testCRDDenyWebhook(f *framework.Framework) {
	ginkgo.By("Creating a custom resource definition that should be denied by the webhook")
	name := fmt.Sprintf("e2e-test-%s-%s-crd", f.BaseName, "deny")
	kind := fmt.Sprintf("E2e-test-%s-%s-crd", f.BaseName, "deny")
	group := fmt.Sprintf("%s.example.com", f.BaseName)
	apiVersions := []apiextensionsv1.CustomResourceDefinitionVersion{
		{
			Name:    "v1",
			Served:  true,
			Storage: true,
			Schema: &apiextensionsv1.CustomResourceValidation{
				OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
					XPreserveUnknownFields: pointer.BoolPtr(true),
					Type:                   "object",
				},
			},
		},
	}

	// Creating a custom resource definition for use by assorted tests.
	config, err := framework.LoadConfig()
	if err != nil {
		framework.Failf("failed to load config: %v", err)
		return
	}
	apiExtensionClient, err := crdclientset.NewForConfig(config)
	if err != nil {
		framework.Failf("failed to initialize apiExtensionClient: %v", err)
		return
	}
	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: name + "s." + group,
			Labels: map[string]string{
				// this label ensures our object is routed to this test's webhook
				f.UniqueName: "true",
				// this is the label the webhook disallows
				"webhook-e2e-test": "webhook-disallow",
			},
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group:    group,
			Versions: apiVersions,
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Singular: name,
				Kind:     kind,
				ListKind: kind + "List",
				Plural:   name + "s",
			},
			Scope: apiextensionsv1.NamespaceScoped,
		},
	}

	// create CRD
	_, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Create(context.TODO(), crd, metav1.CreateOptions{})
	framework.ExpectError(err, "create custom resource definition %s should be denied by webhook", crd.Name)
	expectedErrMsg := "the crd contains unwanted label"
	if !strings.Contains(err.Error(), expectedErrMsg) {
		framework.Failf("expect error contains %q, got %q", expectedErrMsg, err.Error())
	}
}

func labelNamespace(f *framework.Framework, namespace string) {
	client := f.ClientSet

	// Add a unique label to the namespace
	ns, err := client.CoreV1().Namespaces().Get(context.TODO(), namespace, metav1.GetOptions{})
	framework.ExpectNoError(err, "error getting namespace %s", namespace)
	if ns.Labels == nil {
		ns.Labels = map[string]string{}
	}
	ns.Labels[f.UniqueName] = "true"
	_, err = client.CoreV1().Namespaces().Update(context.TODO(), ns, metav1.UpdateOptions{})
	framework.ExpectNoError(err, "error labeling namespace %s", namespace)
}

func registerSlowWebhook(f *framework.Framework, configName string, certCtx *certContext, policy *admissionregistrationv1.FailurePolicyType, timeout *int32, servicePort int32) func() {
	client := f.ClientSet
	ginkgo.By("Registering slow webhook via the AdmissionRegistration API")

	namespace := f.Namespace.Name
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone

	_, err := createValidatingWebhookConfiguration(f, &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: configName,
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name: "allow-configmap-with-delay-webhook.k8s.io",
				Rules: []admissionregistrationv1.RuleWithOperations{{
					Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
					Rule: admissionregistrationv1.Rule{
						APIGroups:   []string{""},
						APIVersions: []string{"v1"},
						Resources:   []string{"configmaps"},
					},
				}},
				ClientConfig: admissionregistrationv1.WebhookClientConfig{
					Service: &admissionregistrationv1.ServiceReference{
						Namespace: namespace,
						Name:      serviceName,
						Path:      strPtr("/always-allow-delay-5s"),
						Port:      pointer.Int32Ptr(servicePort),
					},
					CABundle: certCtx.signingCert,
				},
				// Scope the webhook to just this namespace
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{f.UniqueName: "true"},
				},
				FailurePolicy:           policy,
				TimeoutSeconds:          timeout,
				SideEffects:             &sideEffectsNone,
				AdmissionReviewVersions: []string{"v1", "v1beta1"},
			},
			// Register a webhook that can be probed by marker requests to detect when the configuration is ready.
			newValidatingIsReadyWebhookFixture(f, certCtx, servicePort),
		},
	})
	framework.ExpectNoError(err, "registering slow webhook config %s with namespace %s", configName, namespace)

	err = waitWebhookConfigurationReady(f)
	framework.ExpectNoError(err, "waiting for webhook configuration to be ready")

	return func() {
		client.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(context.TODO(), configName, metav1.DeleteOptions{})
	}
}

func testSlowWebhookTimeoutFailEarly(f *framework.Framework) {
	ginkgo.By("Request fails when timeout (1s) is shorter than slow webhook latency (5s)")
	client := f.ClientSet
	name := "e2e-test-slow-webhook-configmap"
	_, err := client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: name}}, metav1.CreateOptions{})
	framework.ExpectError(err, "create configmap in namespace %s should have timed-out reaching slow webhook", f.Namespace.Name)
	// http timeout message: context deadline exceeded
	// dial timeout message: dial tcp {address}: i/o timeout
	isTimeoutError := strings.Contains(err.Error(), `context deadline exceeded`) || strings.Contains(err.Error(), `timeout`)
	isErrorQueryingWebhook := strings.Contains(err.Error(), `/always-allow-delay-5s?timeout=1s`)
	if !isTimeoutError || !isErrorQueryingWebhook {
		framework.Failf("expect an HTTP/dial timeout error querying the slow webhook, got: %q", err.Error())
	}
}

func testSlowWebhookTimeoutNoError(f *framework.Framework) {
	client := f.ClientSet
	name := "e2e-test-slow-webhook-configmap"
	_, err := client.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), &v1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: name}}, metav1.CreateOptions{})
	gomega.Expect(err).To(gomega.BeNil())
	err = client.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), name, metav1.DeleteOptions{})
	gomega.Expect(err).To(gomega.BeNil())
}

// createAdmissionWebhookMultiVersionTestCRDWithV1Storage creates a new CRD specifically
// for the admissin webhook calling test.
func createAdmissionWebhookMultiVersionTestCRDWithV1Storage(f *framework.Framework, opts ...crd.Option) (*crd.TestCrd, error) {
	group := fmt.Sprintf("%s.example.com", f.BaseName)
	return crd.CreateMultiVersionTestCRD(f, group, append([]crd.Option{func(crd *apiextensionsv1.CustomResourceDefinition) {
		crd.Spec.Versions = []apiextensionsv1.CustomResourceDefinitionVersion{
			{
				Name:    "v1",
				Served:  true,
				Storage: true,
				Schema: &apiextensionsv1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
						XPreserveUnknownFields: pointer.BoolPtr(true),
						Type:                   "object",
					},
				},
			},
			{
				Name:    "v2",
				Served:  true,
				Storage: false,
				Schema: &apiextensionsv1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
						XPreserveUnknownFields: pointer.BoolPtr(true),
						Type:                   "object",
					},
				},
			},
		}
	}}, opts...)...)
}

// servedAPIVersions returns the API versions served by the CRD.
func servedAPIVersions(crd *apiextensionsv1.CustomResourceDefinition) []string {
	ret := []string{}
	for _, v := range crd.Spec.Versions {
		if v.Served {
			ret = append(ret, v.Name)
		}
	}
	return ret
}

// createValidatingWebhookConfiguration ensures the webhook config scopes object or namespace selection
// to avoid interfering with other tests, then creates the config.
func createValidatingWebhookConfiguration(f *framework.Framework, config *admissionregistrationv1.ValidatingWebhookConfiguration) (*admissionregistrationv1.ValidatingWebhookConfiguration, error) {
	for _, webhook := range config.Webhooks {
		if webhook.NamespaceSelector != nil && webhook.NamespaceSelector.MatchLabels[f.UniqueName] == "true" {
			continue
		}
		if webhook.ObjectSelector != nil && webhook.ObjectSelector.MatchLabels[f.UniqueName] == "true" {
			continue
		}
		framework.Failf(`webhook %s in config %s has no namespace or object selector with %s="true", and can interfere with other tests`, webhook.Name, config.Name, f.UniqueName)
	}
	return f.ClientSet.AdmissionregistrationV1().ValidatingWebhookConfigurations().Create(context.TODO(), config, metav1.CreateOptions{})
}

// createMutatingWebhookConfiguration ensures the webhook config scopes object or namespace selection
// to avoid interfering with other tests, then creates the config.
func createMutatingWebhookConfiguration(f *framework.Framework, config *admissionregistrationv1.MutatingWebhookConfiguration) (*admissionregistrationv1.MutatingWebhookConfiguration, error) {
	for _, webhook := range config.Webhooks {
		if webhook.NamespaceSelector != nil && webhook.NamespaceSelector.MatchLabels[f.UniqueName] == "true" {
			continue
		}
		if webhook.ObjectSelector != nil && webhook.ObjectSelector.MatchLabels[f.UniqueName] == "true" {
			continue
		}
		framework.Failf(`webhook %s in config %s has no namespace or object selector with %s="true", and can interfere with other tests`, webhook.Name, config.Name, f.UniqueName)
	}
	return f.ClientSet.AdmissionregistrationV1().MutatingWebhookConfigurations().Create(context.TODO(), config, metav1.CreateOptions{})
}

func newDenyPodWebhookFixture(f *framework.Framework, certCtx *certContext, servicePort int32) admissionregistrationv1.ValidatingWebhook {
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone
	return admissionregistrationv1.ValidatingWebhook{
		Name: "deny-unwanted-pod-container-name-and-label.k8s.io",
		Rules: []admissionregistrationv1.RuleWithOperations{{
			Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
			Rule: admissionregistrationv1.Rule{
				APIGroups:   []string{""},
				APIVersions: []string{"v1"},
				Resources:   []string{"pods"},
			},
		}},
		ClientConfig: admissionregistrationv1.WebhookClientConfig{
			Service: &admissionregistrationv1.ServiceReference{
				Namespace: f.Namespace.Name,
				Name:      serviceName,
				Path:      strPtr("/pods"),
				Port:      pointer.Int32Ptr(servicePort),
			},
			CABundle: certCtx.signingCert,
		},
		SideEffects:             &sideEffectsNone,
		AdmissionReviewVersions: []string{"v1", "v1beta1"},
		// Scope the webhook to just this namespace
		NamespaceSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{f.UniqueName: "true"},
		},
	}
}

func newDenyConfigMapWebhookFixture(f *framework.Framework, certCtx *certContext, servicePort int32) admissionregistrationv1.ValidatingWebhook {
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone
	return admissionregistrationv1.ValidatingWebhook{
		Name: "deny-unwanted-configmap-data.k8s.io",
		Rules: []admissionregistrationv1.RuleWithOperations{{
			Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update, admissionregistrationv1.Delete},
			Rule: admissionregistrationv1.Rule{
				APIGroups:   []string{""},
				APIVersions: []string{"v1"},
				Resources:   []string{"configmaps"},
			},
		}},
		// The webhook skips the namespace that has label "skip-webhook-admission":"yes"
		NamespaceSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{f.UniqueName: "true"},
			MatchExpressions: []metav1.LabelSelectorRequirement{
				{
					Key:      skipNamespaceLabelKey,
					Operator: metav1.LabelSelectorOpNotIn,
					Values:   []string{skipNamespaceLabelValue},
				},
			},
		},
		ClientConfig: admissionregistrationv1.WebhookClientConfig{
			Service: &admissionregistrationv1.ServiceReference{
				Namespace: f.Namespace.Name,
				Name:      serviceName,
				Path:      strPtr("/configmaps"),
				Port:      pointer.Int32Ptr(servicePort),
			},
			CABundle: certCtx.signingCert,
		},
		SideEffects:             &sideEffectsNone,
		AdmissionReviewVersions: []string{"v1", "v1beta1"},
	}
}

func newMutateConfigMapWebhookFixture(f *framework.Framework, certCtx *certContext, stage int, servicePort int32) admissionregistrationv1.MutatingWebhook {
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone
	return admissionregistrationv1.MutatingWebhook{
		Name: fmt.Sprintf("adding-configmap-data-stage-%d.k8s.io", stage),
		Rules: []admissionregistrationv1.RuleWithOperations{{
			Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
			Rule: admissionregistrationv1.Rule{
				APIGroups:   []string{""},
				APIVersions: []string{"v1"},
				Resources:   []string{"configmaps"},
			},
		}},
		ClientConfig: admissionregistrationv1.WebhookClientConfig{
			Service: &admissionregistrationv1.ServiceReference{
				Namespace: f.Namespace.Name,
				Name:      serviceName,
				Path:      strPtr("/mutating-configmaps"),
				Port:      pointer.Int32Ptr(servicePort),
			},
			CABundle: certCtx.signingCert,
		},
		SideEffects:             &sideEffectsNone,
		AdmissionReviewVersions: []string{"v1", "v1beta1"},
		// Scope the webhook to just this namespace
		NamespaceSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{f.UniqueName: "true"},
		},
	}
}

// createWebhookConfigurationReadyNamespace creates a separate namespace for webhook configuration ready markers to
// prevent cross-talk with webhook configurations being tested.
func createWebhookConfigurationReadyNamespace(f *framework.Framework) {
	ns, err := f.ClientSet.CoreV1().Namespaces().Create(context.TODO(), &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:   f.Namespace.Name + "-markers",
			Labels: map[string]string{f.UniqueName + "-markers": "true"},
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating namespace for webhook configuration ready markers")
	f.AddNamespacesToDelete(ns)
}

// waitWebhookConfigurationReady sends "marker" requests until a webhook configuration is ready.
// A webhook created with newValidatingIsReadyWebhookFixture or newMutatingIsReadyWebhookFixture should first be added to
// the webhook configuration.
func waitWebhookConfigurationReady(f *framework.Framework) error {
	cmClient := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name + "-markers")
	return wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		marker := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name: string(uuid.NewUUID()),
				Labels: map[string]string{
					f.UniqueName: "true",
				},
			},
		}
		_, err := cmClient.Create(context.TODO(), marker, metav1.CreateOptions{})
		if err != nil {
			// The always-deny webhook does not provide a reason, so check for the error string we expect
			if strings.Contains(err.Error(), "denied") {
				return true, nil
			}
			return false, err
		}
		// best effort cleanup of markers that are no longer needed
		_ = cmClient.Delete(context.TODO(), marker.GetName(), metav1.DeleteOptions{})
		framework.Logf("Waiting for webhook configuration to be ready...")
		return false, nil
	})
}

// newValidatingIsReadyWebhookFixture creates a validating webhook that can be added to a webhook configuration and then probed
// with "marker" requests via waitWebhookConfigurationReady to wait for a webhook configuration to be ready.
func newValidatingIsReadyWebhookFixture(f *framework.Framework, certCtx *certContext, servicePort int32) admissionregistrationv1.ValidatingWebhook {
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone
	failOpen := admissionregistrationv1.Ignore
	return admissionregistrationv1.ValidatingWebhook{
		Name: "validating-is-webhook-configuration-ready.k8s.io",
		Rules: []admissionregistrationv1.RuleWithOperations{{
			Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
			Rule: admissionregistrationv1.Rule{
				APIGroups:   []string{""},
				APIVersions: []string{"v1"},
				Resources:   []string{"configmaps"},
			},
		}},
		ClientConfig: admissionregistrationv1.WebhookClientConfig{
			Service: &admissionregistrationv1.ServiceReference{
				Namespace: f.Namespace.Name,
				Name:      serviceName,
				Path:      strPtr("/always-deny"),
				Port:      pointer.Int32Ptr(servicePort),
			},
			CABundle: certCtx.signingCert,
		},
		// network failures while the service network routing is being set up should be ignored by the marker
		FailurePolicy:           &failOpen,
		SideEffects:             &sideEffectsNone,
		AdmissionReviewVersions: []string{"v1", "v1beta1"},
		// Scope the webhook to just the markers namespace
		NamespaceSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{f.UniqueName + "-markers": "true"},
		},
		// appease createValidatingWebhookConfiguration isolation requirements
		ObjectSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{f.UniqueName: "true"},
		},
	}
}

// newMutatingIsReadyWebhookFixture creates a mutating webhook that can be added to a webhook configuration and then probed
// with "marker" requests via waitWebhookConfigurationReady to wait for a webhook configuration to be ready.
func newMutatingIsReadyWebhookFixture(f *framework.Framework, certCtx *certContext, servicePort int32) admissionregistrationv1.MutatingWebhook {
	sideEffectsNone := admissionregistrationv1.SideEffectClassNone
	failOpen := admissionregistrationv1.Ignore
	return admissionregistrationv1.MutatingWebhook{
		Name: "mutating-is-webhook-configuration-ready.k8s.io",
		Rules: []admissionregistrationv1.RuleWithOperations{{
			Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create},
			Rule: admissionregistrationv1.Rule{
				APIGroups:   []string{""},
				APIVersions: []string{"v1"},
				Resources:   []string{"configmaps"},
			},
		}},
		ClientConfig: admissionregistrationv1.WebhookClientConfig{
			Service: &admissionregistrationv1.ServiceReference{
				Namespace: f.Namespace.Name,
				Name:      serviceName,
				Path:      strPtr("/always-deny"),
				Port:      pointer.Int32Ptr(servicePort),
			},
			CABundle: certCtx.signingCert,
		},
		// network failures while the service network routing is being set up should be ignored by the marker
		FailurePolicy:           &failOpen,
		SideEffects:             &sideEffectsNone,
		AdmissionReviewVersions: []string{"v1", "v1beta1"},
		// Scope the webhook to just the markers namespace
		NamespaceSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{f.UniqueName + "-markers": "true"},
		},
		// appease createMutatingWebhookConfiguration isolation requirements
		ObjectSelector: &metav1.LabelSelector{
			MatchLabels: map[string]string{f.UniqueName: "true"},
		},
	}
}
