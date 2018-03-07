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
	"bufio"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	apiv1 "k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/apis/audit/v1beta1"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/evanphx/json-patch"
	. "github.com/onsi/ginkgo"
)

var (
	watchTestTimeout int64 = 1
	auditTestUser          = "kubecfg"

	crd          = testserver.NewRandomNameCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
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
		apiExtensionClient, err := clientset.NewForConfig(config)
		framework.ExpectNoError(err, "failed to initialize apiExtensionClient")

		testCases := []struct {
			action func()
			events []auditEvent
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
								Image: framework.GetPauseImageName(f.ClientSet),
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
				[]auditEvent{
					{
						v1beta1.LevelRequestResponse,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/pods", namespace),
						"create",
						201,
						auditTestUser,
						"pods",
						namespace,
						true,
						true,
					}, {
						v1beta1.LevelRequest,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/pods/audit-pod", namespace),
						"get",
						200,
						auditTestUser,
						"pods",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelRequest,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/pods", namespace),
						"list",
						200,
						auditTestUser,
						"pods",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelRequest,
						v1beta1.StageResponseStarted,
						fmt.Sprintf("/api/v1/namespaces/%s/pods?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						"watch",
						200,
						auditTestUser,
						"pods",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelRequest,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/pods?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						"watch",
						200,
						auditTestUser,
						"pods",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelRequestResponse,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/pods/audit-pod", namespace),
						"update",
						200,
						auditTestUser,
						"pods",
						namespace,
						true,
						true,
					}, {
						v1beta1.LevelRequestResponse,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/pods/audit-pod", namespace),
						"patch",
						200,
						auditTestUser,
						"pods",
						namespace,
						true,
						true,
					}, {
						v1beta1.LevelRequestResponse,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/pods/audit-pod", namespace),
						"delete",
						200,
						auditTestUser,
						"pods",
						namespace,
						true,
						true,
					},
				},
			},
			// Create, get, update, patch, delete, list, watch deployments.
			{
				func() {
					podLabels := map[string]string{"name": "audit-deployment-pod"}
					d := framework.NewDeployment("audit-deployment", int32(1), podLabels, "redis", imageutils.GetE2EImage(imageutils.Redis), extensions.RecreateDeploymentStrategyType)

					_, err := f.ClientSet.ExtensionsV1beta1().Deployments(namespace).Create(d)
					framework.ExpectNoError(err, "failed to create audit-deployment")

					_, err = f.ClientSet.ExtensionsV1beta1().Deployments(namespace).Get(d.Name, metav1.GetOptions{})
					framework.ExpectNoError(err, "failed to get audit-deployment")

					deploymentChan, err := f.ClientSet.ExtensionsV1beta1().Deployments(namespace).Watch(watchOptions)
					framework.ExpectNoError(err, "failed to create watch for deployments")
					for range deploymentChan.ResultChan() {
					}

					_, err = f.ClientSet.ExtensionsV1beta1().Deployments(namespace).Update(d)
					framework.ExpectNoError(err, "failed to update audit-deployment")

					_, err = f.ClientSet.ExtensionsV1beta1().Deployments(namespace).Patch(d.Name, types.JSONPatchType, patch)
					framework.ExpectNoError(err, "failed to patch deployment")

					_, err = f.ClientSet.ExtensionsV1beta1().Deployments(namespace).List(metav1.ListOptions{})
					framework.ExpectNoError(err, "failed to create list deployments")

					err = f.ClientSet.ExtensionsV1beta1().Deployments(namespace).Delete("audit-deployment", &metav1.DeleteOptions{})
					framework.ExpectNoError(err, "failed to delete deployments")
				},
				[]auditEvent{
					{
						v1beta1.LevelRequestResponse,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/apis/extensions/v1beta1/namespaces/%s/deployments", namespace),
						"create",
						201,
						auditTestUser,
						"deployments",
						namespace,
						true,
						true,
					}, {
						v1beta1.LevelRequest,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/apis/extensions/v1beta1/namespaces/%s/deployments/audit-deployment", namespace),
						"get",
						200,
						auditTestUser,
						"deployments",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelRequest,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/apis/extensions/v1beta1/namespaces/%s/deployments", namespace),
						"list",
						200,
						auditTestUser,
						"deployments",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelRequest,
						v1beta1.StageResponseStarted,
						fmt.Sprintf("/apis/extensions/v1beta1/namespaces/%s/deployments?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						"watch",
						200,
						auditTestUser,
						"deployments",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelRequest,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/apis/extensions/v1beta1/namespaces/%s/deployments?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						"watch",
						200,
						auditTestUser,
						"deployments",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelRequestResponse,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/apis/extensions/v1beta1/namespaces/%s/deployments/audit-deployment", namespace),
						"update",
						200,
						auditTestUser,
						"deployments",
						namespace,
						true,
						true,
					}, {
						v1beta1.LevelRequestResponse,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/apis/extensions/v1beta1/namespaces/%s/deployments/audit-deployment", namespace),
						"patch",
						200,
						auditTestUser,
						"deployments",
						namespace,
						true,
						true,
					}, {
						v1beta1.LevelRequestResponse,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/apis/extensions/v1beta1/namespaces/%s/deployments/audit-deployment", namespace),
						"delete",
						200,
						auditTestUser,
						"deployments",
						namespace,
						true,
						true,
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
				[]auditEvent{
					{
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/configmaps", namespace),
						"create",
						201,
						auditTestUser,
						"configmaps",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/configmaps/audit-configmap", namespace),
						"get",
						200,
						auditTestUser,
						"configmaps",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/configmaps", namespace),
						"list",
						200,
						auditTestUser,
						"configmaps",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseStarted,
						fmt.Sprintf("/api/v1/namespaces/%s/configmaps?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						"watch",
						200,
						auditTestUser,
						"configmaps",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/configmaps?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						"watch",
						200,
						auditTestUser,
						"configmaps",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/configmaps/audit-configmap", namespace),
						"update",
						200,
						auditTestUser,
						"configmaps",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/configmaps/audit-configmap", namespace),
						"patch",
						200,
						auditTestUser,
						"configmaps",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/configmaps/audit-configmap", namespace),
						"delete",
						200,
						auditTestUser,
						"configmaps",
						namespace,
						false,
						false,
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
				[]auditEvent{
					{
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/secrets", namespace),
						"create",
						201,
						auditTestUser,
						"secrets",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/secrets/audit-secret", namespace),
						"get",
						200,
						auditTestUser,
						"secrets",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/secrets", namespace),
						"list",
						200,
						auditTestUser,
						"secrets",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseStarted,
						fmt.Sprintf("/api/v1/namespaces/%s/secrets?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						"watch",
						200,
						auditTestUser,
						"secrets",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/secrets?timeoutSeconds=%d&watch=true", namespace, watchTestTimeout),
						"watch",
						200,
						auditTestUser,
						"secrets",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/secrets/audit-secret", namespace),
						"update",
						200,
						auditTestUser,
						"secrets",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/secrets/audit-secret", namespace),
						"patch",
						200,
						auditTestUser,
						"secrets",
						namespace,
						false,
						false,
					}, {
						v1beta1.LevelMetadata,
						v1beta1.StageResponseComplete,
						fmt.Sprintf("/api/v1/namespaces/%s/secrets/audit-secret", namespace),
						"delete",
						200,
						auditTestUser,
						"secrets",
						namespace,
						false,
						false,
					},
				},
			},
			// Create and delete custom resource definition.
			{
				func() {
					_, err = testserver.CreateNewCustomResourceDefinition(crd, apiExtensionClient, f.ClientPool)
					framework.ExpectNoError(err, "failed to create custom resource definition")
					testserver.DeleteCustomResourceDefinition(crd, apiExtensionClient)
				},
				[]auditEvent{
					{
						level:          v1beta1.LevelRequestResponse,
						stage:          v1beta1.StageResponseComplete,
						requestURI:     "/apis/apiextensions.k8s.io/v1beta1/customresourcedefinitions",
						verb:           "create",
						code:           201,
						user:           auditTestUser,
						resource:       "customresourcedefinitions",
						requestObject:  true,
						responseObject: true,
					}, {
						level:          v1beta1.LevelMetadata,
						stage:          v1beta1.StageResponseComplete,
						requestURI:     fmt.Sprintf("/apis/%s/v1beta1/%s", crdNamespace, crdName),
						verb:           "create",
						code:           201,
						user:           auditTestUser,
						resource:       crdName,
						requestObject:  false,
						responseObject: false,
					}, {
						level:          v1beta1.LevelRequestResponse,
						stage:          v1beta1.StageResponseComplete,
						requestURI:     fmt.Sprintf("/apis/apiextensions.k8s.io/v1beta1/customresourcedefinitions/%s", crd.Name),
						verb:           "delete",
						code:           200,
						user:           auditTestUser,
						resource:       "customresourcedefinitions",
						requestObject:  false,
						responseObject: true,
					}, {
						level:          v1beta1.LevelMetadata,
						stage:          v1beta1.StageResponseComplete,
						requestURI:     fmt.Sprintf("/apis/%s/v1beta1/%s/setup-instance", crdNamespace, crdName),
						verb:           "delete",
						code:           200,
						user:           auditTestUser,
						resource:       crdName,
						requestObject:  false,
						responseObject: false,
					},
				},
			},
		}

		expectedEvents := []auditEvent{}
		for _, t := range testCases {
			t.action()
			expectedEvents = append(expectedEvents, t.events...)
		}

		// The default flush timeout is 30 seconds, therefore it should be enough to retry once
		// to find all expected events. However, we're waiting for 5 minutes to avoid flakes.
		pollingInterval := 30 * time.Second
		pollingTimeout := 5 * time.Minute
		err = wait.Poll(pollingInterval, pollingTimeout, func() (bool, error) {
			ok, err := checkAuditLines(f, expectedEvents)
			if err != nil {
				framework.Logf("Failed to observe audit events: %v", err)
			}
			return ok, nil
		})
		framework.ExpectNoError(err, "after %v failed to observe audit events", pollingTimeout)
	})
})

type auditEvent struct {
	level          v1beta1.Level
	stage          v1beta1.Stage
	requestURI     string
	verb           string
	code           int32
	user           string
	resource       string
	namespace      string
	requestObject  bool
	responseObject bool
}

// Search the audit log for the expected audit lines.
func checkAuditLines(f *framework.Framework, expected []auditEvent) (bool, error) {
	expectations := map[auditEvent]bool{}
	for _, event := range expected {
		expectations[event] = false
	}

	// Fetch the log stream.
	stream, err := f.ClientSet.CoreV1().RESTClient().Get().AbsPath("/logs/kube-apiserver-audit.log").Stream()
	if err != nil {
		return false, err
	}
	defer stream.Close()

	scanner := bufio.NewScanner(stream)
	for scanner.Scan() {
		line := scanner.Text()
		event, err := parseAuditLine(line)
		if err != nil {
			return false, err
		}

		// If the event was expected, mark it as found.
		if _, found := expectations[event]; found {
			expectations[event] = true
		}
	}
	if err := scanner.Err(); err != nil {
		return false, err
	}

	noneMissing := true
	for event, found := range expectations {
		if !found {
			framework.Logf("Event %#v not found!", event)
		}
		noneMissing = noneMissing && found
	}
	return noneMissing, nil
}

func parseAuditLine(line string) (auditEvent, error) {
	var e v1beta1.Event
	if err := json.Unmarshal([]byte(line), &e); err != nil {
		return auditEvent{}, err
	}
	event := auditEvent{
		level:      e.Level,
		stage:      e.Stage,
		requestURI: e.RequestURI,
		verb:       e.Verb,
		user:       e.User.Username,
	}
	if e.ObjectRef != nil {
		event.namespace = e.ObjectRef.Namespace
		event.resource = e.ObjectRef.Resource
	}
	if e.ResponseStatus != nil {
		event.code = e.ResponseStatus.Code
	}
	if e.ResponseObject != nil {
		event.responseObject = true
	}
	if e.RequestObject != nil {
		event.requestObject = true
	}
	return event, nil
}
