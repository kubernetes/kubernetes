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
	"crypto/rand"
	"encoding/json"
	"fmt"
	"math/big"
	"net"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	unstructuredv1 "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/discovery"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
	aggregatorclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	samplev1alpha1 "k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1"
	"k8s.io/utils/pointer"

	"github.com/onsi/ginkgo/v2"
)

const (
	aggregatorServicePort = 7443

	apiServiceRetryPeriod  = 1 * time.Second
	apiServiceRetryTimeout = 1 * time.Minute
)

var _ = SIGDescribe("Aggregator", func() {
	var ns string
	var c clientset.Interface
	var aggrclient *aggregatorclient.Clientset

	// BeforeEachs run in LIFO order, AfterEachs run in FIFO order.
	// We want cleanTest to happen before the namespace cleanup AfterEach
	// inserted by NewDefaultFramework, so we put this AfterEach in front
	// of NewDefaultFramework.
	ginkgo.AfterEach(func() {
		cleanTest(c, aggrclient, ns)
	})

	f := framework.NewDefaultFramework("aggregator")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	// We want namespace initialization BeforeEach inserted by
	// NewDefaultFramework to happen before this, so we put this BeforeEach
	// after NewDefaultFramework.
	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name

		if aggrclient == nil {
			config, err := framework.LoadConfig()
			if err != nil {
				framework.Failf("could not load config: %v", err)
			}
			aggrclient, err = aggregatorclient.NewForConfig(config)
			if err != nil {
				framework.Failf("could not create aggregator client: %v", err)
			}
		}
	})

	/*
		    Release: v1.17, v1.21
		    Testname: aggregator-supports-the-sample-apiserver
		    Description: Ensure that the sample-apiserver code from 1.17 and compiled against 1.17
			will work on the current Aggregator/API-Server.
	*/
	framework.ConformanceIt("Should be able to support the 1.17 Sample API Server using the current Aggregator", func() {
		// Testing a 1.17 version of the sample-apiserver
		TestSampleAPIServer(f, aggrclient, imageutils.GetE2EImage(imageutils.APIServer))
	})

	ginkgo.It("should manage the lifecycle of an APIService", func() {

		ns := f.Namespace.Name
		framework.Logf("ns: %v", ns)

		subDomain := "e2e-" + utilrand.String(5)
		apiServiceGroup := subDomain + ".example.com"
		label := map[string]string{"e2e": subDomain}
		labelSelector := labels.SelectorFromSet(label).String()

		apiServiceName := "v1alpha1." + apiServiceGroup
		apiServiceClient := aggrclient.ApiregistrationV1().APIServices()
		certCtx := setupServerCert(ns, "e2e-api")

		ginkgo.By(fmt.Sprintf("Create APIService %s", apiServiceName))
		_, err := apiServiceClient.Create(context.TODO(), &apiregistrationv1.APIService{
			ObjectMeta: metav1.ObjectMeta{
				Name:   apiServiceName,
				Labels: label,
			},
			Spec: apiregistrationv1.APIServiceSpec{
				Service: &apiregistrationv1.ServiceReference{
					Namespace: ns,
					Name:      "e2e-api",
					Port:      pointer.Int32Ptr(aggregatorServicePort),
				},
				Group:                apiServiceGroup,
				Version:              "v1alpha1",
				CABundle:             certCtx.signingCert,
				GroupPriorityMinimum: 2000,
				VersionPriority:      200,
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "creating apiService %s, namespace %s", apiServiceName, ns)

		_, err = apiServiceClient.List(context.TODO(), metav1.ListOptions{LabelSelector: labelSelector})
		framework.ExpectNoError(err, "failed to list API Services")

		ginkgo.By("Confirm that the generated APIService has been created")
		err = wait.PollImmediate(apiServiceRetryPeriod, apiServiceRetryTimeout, checkApiServiceListQuantity(aggrclient, labelSelector, 1))
		framework.ExpectNoError(err, "failed to count the required APIServices")

		ginkgo.By(fmt.Sprintf("Update status for APIService %s", apiServiceName))
		var statusToUpdate, updatedStatus *apiregistrationv1.APIService

		updatedStatusConditions := apiregistrationv1.APIServiceCondition{
			Type:    "StatusUpdate",
			Status:  "True",
			Reason:  "E2E",
			Message: "Set from e2e test",
		}

		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			statusToUpdate, err = apiServiceClient.Get(context.TODO(), apiServiceName, metav1.GetOptions{})
			framework.ExpectNoError(err, "Unable to retrieve api service %s", apiServiceName)

			statusToUpdate.Status.Conditions = append(statusToUpdate.Status.Conditions, updatedStatusConditions)

			updatedStatus, err = apiServiceClient.UpdateStatus(context.TODO(), statusToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "Failed to update status. %v", err)
		framework.Logf("updatedStatus.Conditions: %#v", updatedStatus.Status.Conditions)

		ginkgo.By("Confirm that the generated APIService has an updated status")
		err = wait.PollImmediate(apiServiceRetryPeriod, apiServiceRetryTimeout, checkApiServiceStatus(aggrclient, apiServiceName, updatedStatusConditions))
		framework.ExpectNoError(err, "failed to locate the required APIService status")

		ginkgo.By(fmt.Sprintf("Patching status for APIService %s", apiServiceName))

		patchedStatusConditions := apiregistrationv1.APIServiceCondition{
			Type:    "StatusPatched",
			Status:  "True",
			Reason:  "E2E",
			Message: "Set from e2e test",
		}

		payload := []byte(`{"status":{"conditions":[{"type":"StatusPatched","status":"True","reason":"E2E","message":"Set from e2e test"}]}}`)

		patchedApiService, err := apiServiceClient.Patch(context.TODO(), apiServiceName, types.MergePatchType, payload, metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err, "Failed to patch status. %v", err)
		framework.Logf("Patched status conditions: %#v", patchedApiService.Status.Conditions)

		ginkgo.By("Confirm that the generated APIService has a patched status")
		err = wait.PollImmediate(apiServiceRetryPeriod, apiServiceRetryTimeout, checkApiServiceStatus(aggrclient, apiServiceName, patchedStatusConditions))
		framework.ExpectNoError(err, "failed to locate the required APIService status")

		ginkgo.By(fmt.Sprintf("Replace APIService %s", apiServiceName))
		var updatedApiService *apiregistrationv1.APIService

		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			patchedApiService, err = apiServiceClient.Get(context.TODO(), apiServiceName, metav1.GetOptions{})
			framework.ExpectNoError(err, "Unable to get APIService %s", apiServiceName)
			patchedApiService.Labels = map[string]string{
				apiServiceGroup: "updated",
			}
			updatedApiService, err = apiServiceClient.Update(context.TODO(), patchedApiService, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err)
		framework.ExpectEqual(updatedApiService.Labels[apiServiceGroup], "updated", "updated object should have the applied label")
		framework.Logf("Found updated apiService label for %q", apiServiceName)

		ginkgo.By(fmt.Sprintf("DeleteCollection APIService %s via labelSelector: %s", apiServiceName, labelSelector))

		err = aggrclient.ApiregistrationV1().APIServices().DeleteCollection(context.TODO(),
			metav1.DeleteOptions{GracePeriodSeconds: pointer.Int64(1)},
			metav1.ListOptions{LabelSelector: labelSelector})
		framework.ExpectNoError(err, "Unable to delete apiservice %s", apiServiceName)

		ginkgo.By("Confirm that the generated APIService has been deleted")
		err = wait.PollImmediate(apiServiceRetryPeriod, apiServiceRetryTimeout, checkApiServiceListQuantity(aggrclient, labelSelector, 0))
		framework.ExpectNoError(err, "failed to count the required APIServices")
		framework.Logf("APIService %s has been deleted.", apiServiceName)
	})
})

func cleanTest(client clientset.Interface, aggrclient *aggregatorclient.Clientset, namespace string) {
	// delete the APIService first to avoid causing discovery errors
	_ = aggrclient.ApiregistrationV1().APIServices().Delete(context.TODO(), "v1alpha1.wardle.example.com", metav1.DeleteOptions{})

	_ = client.AppsV1().Deployments(namespace).Delete(context.TODO(), "sample-apiserver-deployment", metav1.DeleteOptions{})
	_ = client.CoreV1().Secrets(namespace).Delete(context.TODO(), "sample-apiserver-secret", metav1.DeleteOptions{})
	_ = client.CoreV1().Services(namespace).Delete(context.TODO(), "sample-api", metav1.DeleteOptions{})
	_ = client.CoreV1().ServiceAccounts(namespace).Delete(context.TODO(), "sample-apiserver", metav1.DeleteOptions{})
	_ = client.RbacV1().RoleBindings("kube-system").Delete(context.TODO(), "wardler-auth-reader", metav1.DeleteOptions{})
	_ = client.RbacV1().ClusterRoleBindings().Delete(context.TODO(), "wardler:"+namespace+":auth-delegator", metav1.DeleteOptions{})
	_ = client.RbacV1().ClusterRoles().Delete(context.TODO(), "sample-apiserver-reader", metav1.DeleteOptions{})
	_ = client.RbacV1().ClusterRoleBindings().Delete(context.TODO(), "wardler:"+namespace+":sample-apiserver-reader", metav1.DeleteOptions{})
}

// TestSampleAPIServer is a basic test if the sample-apiserver code from 1.10 and compiled against 1.10
// will work on the current Aggregator/API-Server.
func TestSampleAPIServer(f *framework.Framework, aggrclient *aggregatorclient.Clientset, image string) {
	ginkgo.By("Registering the sample API server.")
	client := f.ClientSet
	restClient := client.Discovery().RESTClient()

	namespace := f.Namespace.Name
	certCtx := setupServerCert(namespace, "sample-api")

	// kubectl create -f namespace.yaml
	// NOTE: aggregated apis should generally be set up in their own namespace. As the test framework is setting up a new namespace, we are just using that.

	// kubectl create -f secret.yaml
	secretName := "sample-apiserver-secret"
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
	_, err := client.CoreV1().Secrets(namespace).Create(context.TODO(), secret, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating secret %s in namespace %s", secretName, namespace)

	// kubectl create -f clusterrole.yaml
	_, err = client.RbacV1().ClusterRoles().Create(context.TODO(), &rbacv1.ClusterRole{

		ObjectMeta: metav1.ObjectMeta{Name: "sample-apiserver-reader"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch").Groups("").Resources("namespaces").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "watch").Groups("admissionregistration.k8s.io").Resources("*").RuleOrDie(),
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating cluster role %s", "sample-apiserver-reader")

	_, err = client.RbacV1().ClusterRoleBindings().Create(context.TODO(), &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "wardler:" + namespace + ":sample-apiserver-reader",
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "sample-apiserver-reader",
		},
		Subjects: []rbacv1.Subject{
			{
				APIGroup:  "",
				Kind:      "ServiceAccount",
				Name:      "default",
				Namespace: namespace,
			},
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating cluster role binding %s", "wardler:"+namespace+":sample-apiserver-reader")

	// kubectl create -f authDelegator.yaml
	_, err = client.RbacV1().ClusterRoleBindings().Create(context.TODO(), &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "wardler:" + namespace + ":auth-delegator",
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "system:auth-delegator",
		},
		Subjects: []rbacv1.Subject{
			{
				APIGroup:  "",
				Kind:      "ServiceAccount",
				Name:      "default",
				Namespace: namespace,
			},
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating cluster role binding %s", "wardler:"+namespace+":auth-delegator")

	// kubectl create -f deploy.yaml
	deploymentName := "sample-apiserver-deployment"
	etcdImage := imageutils.GetE2EImage(imageutils.Etcd)
	podLabels := map[string]string{"app": "sample-apiserver", "apiserver": "true"}
	replicas := int32(1)
	etcdLocalhostAddress := "127.0.0.1"
	if framework.TestContext.ClusterIsIPv6() {
		etcdLocalhostAddress = "::1"
	}
	etcdURL := fmt.Sprintf("http://%s", net.JoinHostPort(etcdLocalhostAddress, "2379"))

	mounts := []v1.VolumeMount{
		{
			Name:      "apiserver-certs",
			ReadOnly:  true,
			MountPath: "/apiserver.local.config/certificates",
		},
	}
	volumes := []v1.Volume{
		{
			Name: "apiserver-certs",
			VolumeSource: v1.VolumeSource{
				Secret: &v1.SecretVolumeSource{SecretName: secretName},
			},
		},
	}
	containers := []v1.Container{
		{
			Name:         "sample-apiserver",
			VolumeMounts: mounts,
			Args: []string{
				fmt.Sprintf("--etcd-servers=%s", etcdURL),
				"--tls-cert-file=/apiserver.local.config/certificates/tls.crt",
				"--tls-private-key-file=/apiserver.local.config/certificates/tls.key",
				"--audit-log-path=-",
				"--audit-log-maxage=0",
				"--audit-log-maxbackup=0",
			},
			Image: image,
			ReadinessProbe: &v1.Probe{
				ProbeHandler: v1.ProbeHandler{
					HTTPGet: &v1.HTTPGetAction{
						Scheme: v1.URISchemeHTTPS,
						Port:   intstr.FromInt(443),
						Path:   "/readyz",
					},
				},
				InitialDelaySeconds: 20,
				PeriodSeconds:       1,
				SuccessThreshold:    1,
				FailureThreshold:    3,
			},
		},
		{
			Name:  "etcd",
			Image: etcdImage,
			Command: []string{
				"/usr/local/bin/etcd",
				"--listen-client-urls",
				etcdURL,
				"--advertise-client-urls",
				etcdURL,
			},
		},
	}
	d := e2edeployment.NewDeployment(deploymentName, replicas, podLabels, "", "", appsv1.RollingUpdateDeploymentStrategyType)
	d.Spec.Template.Spec.Containers = containers
	d.Spec.Template.Spec.Volumes = volumes

	deployment, err := client.AppsV1().Deployments(namespace).Create(context.TODO(), d, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating deployment %s in namespace %s", deploymentName, namespace)

	err = e2edeployment.WaitForDeploymentRevisionAndImage(client, namespace, deploymentName, "1", image)
	framework.ExpectNoError(err, "waiting for the deployment of image %s in %s in %s to complete", image, deploymentName, namespace)

	err = e2edeployment.WaitForDeploymentRevisionAndImage(client, namespace, deploymentName, "1", etcdImage)
	framework.ExpectNoError(err, "waiting for the deployment of image %s in %s in %s to complete", etcdImage, deploymentName, namespace)

	// kubectl create -f service.yaml
	serviceLabels := map[string]string{"apiserver": "true"}
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      "sample-api",
			Labels:    map[string]string{"test": "aggregator"},
		},
		Spec: v1.ServiceSpec{
			Selector: serviceLabels,
			Ports: []v1.ServicePort{
				{
					Protocol:   "TCP",
					Port:       aggregatorServicePort,
					TargetPort: intstr.FromInt(443),
				},
			},
		},
	}
	_, err = client.CoreV1().Services(namespace).Create(context.TODO(), service, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating service %s in namespace %s", "sample-api", namespace)

	// kubectl create -f serviceAccount.yaml
	sa := &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "sample-apiserver"}}
	_, err = client.CoreV1().ServiceAccounts(namespace).Create(context.TODO(), sa, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating service account %s in namespace %s", "sample-apiserver", namespace)

	// kubectl create -f auth-reader.yaml
	_, err = client.RbacV1().RoleBindings("kube-system").Create(context.TODO(), &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "wardler-auth-reader",
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
	framework.ExpectNoError(err, "creating role binding %s in namespace %s", "wardler-auth-reader", "kube-system")

	// Wait for the extension apiserver to be up and healthy
	// kubectl get deployments -n <aggregated-api-namespace> && status == Running
	// NOTE: aggregated apis should generally be set up in their own namespace (<aggregated-api-namespace>). As the test framework
	// is setting up a new namespace, we are just using that.
	err = e2edeployment.WaitForDeploymentComplete(client, deployment)
	framework.ExpectNoError(err, "deploying extension apiserver in namespace %s", namespace)

	// kubectl create -f apiservice.yaml
	_, err = aggrclient.ApiregistrationV1().APIServices().Create(context.TODO(), &apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: "v1alpha1.wardle.example.com"},
		Spec: apiregistrationv1.APIServiceSpec{
			Service: &apiregistrationv1.ServiceReference{
				Namespace: namespace,
				Name:      "sample-api",
				Port:      pointer.Int32Ptr(aggregatorServicePort),
			},
			Group:                "wardle.example.com",
			Version:              "v1alpha1",
			CABundle:             certCtx.signingCert,
			GroupPriorityMinimum: 2000,
			VersionPriority:      200,
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err, "creating apiservice %s", "v1alpha1.wardle.example.com")

	var (
		currentAPIService *apiregistrationv1.APIService
		currentPods       *v1.PodList
	)

	err = pollTimed(100*time.Millisecond, 60*time.Second, func() (bool, error) {

		currentAPIService, _ = aggrclient.ApiregistrationV1().APIServices().Get(context.TODO(), "v1alpha1.wardle.example.com", metav1.GetOptions{})
		currentPods, _ = client.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{})

		request := restClient.Get().AbsPath("/apis/wardle.example.com/v1alpha1/namespaces/default/flunders")
		request.SetHeader("Accept", "application/json")
		_, err := request.DoRaw(context.TODO())
		if err != nil {
			status, ok := err.(*apierrors.StatusError)
			if !ok {
				return false, err
			}
			if status.Status().Code == 403 || status.Status().Code == 503 {
				return false, nil
			}
			if status.Status().Code == 404 && strings.HasPrefix(err.Error(), "the server could not find the requested resource") {
				return false, nil
			}
			return false, err
		}
		return true, nil
	}, "Waited %s for the sample-apiserver to be ready to handle requests.")
	if err != nil {
		currentAPIServiceJSON, _ := json.Marshal(currentAPIService)
		framework.Logf("current APIService: %s", string(currentAPIServiceJSON))

		currentPodsJSON, _ := json.Marshal(currentPods)
		framework.Logf("current pods: %s", string(currentPodsJSON))

		if currentPods != nil {
			for _, pod := range currentPods.Items {
				for _, container := range pod.Spec.Containers {
					logs, err := e2epod.GetPodLogs(client, namespace, pod.Name, container.Name)
					framework.Logf("logs of %s/%s (error: %v): %s", pod.Name, container.Name, err, logs)
				}
			}
		}
	}
	framework.ExpectNoError(err, "gave up waiting for apiservice wardle to come up successfully")

	flunderName := generateFlunderName("rest-flunder")

	// kubectl create -f flunders-1.yaml -v 9
	// curl -k -v -XPOST https://localhost/apis/wardle.example.com/v1alpha1/namespaces/default/flunders
	// Request Body: {"apiVersion":"wardle.example.com/v1alpha1","kind":"Flunder","metadata":{"labels":{"sample-label":"true"},"name":"test-flunder","namespace":"default"}}
	flunder := `{"apiVersion":"wardle.example.com/v1alpha1","kind":"Flunder","metadata":{"labels":{"sample-label":"true"},"name":"` + flunderName + `","namespace":"default"}}`
	result := restClient.Post().AbsPath("/apis/wardle.example.com/v1alpha1/namespaces/default/flunders").Body([]byte(flunder)).SetHeader("Accept", "application/json").Do(context.TODO())
	framework.ExpectNoError(result.Error(), "creating a new flunders resource")
	var statusCode int
	result.StatusCode(&statusCode)
	if statusCode != 201 {
		framework.Failf("Flunders client creation response was status %d, not 201", statusCode)
	}
	u := &unstructured.Unstructured{}
	if err := result.Into(u); err != nil {
		framework.ExpectNoError(err, "reading created response")
	}
	framework.ExpectEqual(u.GetAPIVersion(), "wardle.example.com/v1alpha1")
	framework.ExpectEqual(u.GetKind(), "Flunder")
	framework.ExpectEqual(u.GetName(), flunderName)

	pods, err := client.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{})
	framework.ExpectNoError(err, "getting pods for flunders service")

	// kubectl get flunders -v 9
	// curl -k -v -XGET https://localhost/apis/wardle.example.com/v1alpha1/namespaces/default/flunders
	contents, err := restClient.Get().AbsPath("/apis/wardle.example.com/v1alpha1/namespaces/default/flunders").SetHeader("Accept", "application/json").DoRaw(context.TODO())
	framework.ExpectNoError(err, "attempting to get a newly created flunders resource")
	var flundersList samplev1alpha1.FlunderList
	err = json.Unmarshal(contents, &flundersList)
	validateErrorWithDebugInfo(f, err, pods, "Error in unmarshalling %T response from server %s", contents, "/apis/wardle.example.com/v1alpha1")
	if len(flundersList.Items) != 1 {
		framework.Failf("failed to get back the correct flunders list %v", flundersList)
	}

	// kubectl delete flunder test-flunder -v 9
	// curl -k -v -XDELETE  https://35.193.112.40/apis/wardle.example.com/v1alpha1/namespaces/default/flunders/test-flunder
	_, err = restClient.Delete().AbsPath("/apis/wardle.example.com/v1alpha1/namespaces/default/flunders/" + flunderName).DoRaw(context.TODO())
	validateErrorWithDebugInfo(f, err, pods, "attempting to delete a newly created flunders(%v) resource", flundersList.Items)

	// kubectl get flunders -v 9
	// curl -k -v -XGET https://localhost/apis/wardle.example.com/v1alpha1/namespaces/default/flunders
	contents, err = restClient.Get().AbsPath("/apis/wardle.example.com/v1alpha1/namespaces/default/flunders").SetHeader("Accept", "application/json").DoRaw(context.TODO())
	framework.ExpectNoError(err, "confirming delete of a newly created flunders resource")
	err = json.Unmarshal(contents, &flundersList)
	validateErrorWithDebugInfo(f, err, pods, "Error in unmarshalling %T response from server %s", contents, "/apis/wardle.example.com/v1alpha1")
	if len(flundersList.Items) != 0 {
		framework.Failf("failed to get back the correct deleted flunders list %v", flundersList)
	}

	flunderName = generateFlunderName("dynamic-flunder")

	// Rerun the Create/List/Delete tests using the Dynamic client.
	resources, discoveryErr := client.Discovery().ServerPreferredNamespacedResources()
	groupVersionResources, err := discovery.GroupVersionResources(resources)
	framework.ExpectNoError(err, "getting group version resources for dynamic client")
	gvr := schema.GroupVersionResource{Group: "wardle.example.com", Version: "v1alpha1", Resource: "flunders"}
	_, ok := groupVersionResources[gvr]
	if !ok {
		framework.Failf("could not find group version resource for dynamic client and wardle/flunders (discovery error: %v, discovery results: %#v)", discoveryErr, groupVersionResources)
	}
	dynamicClient := f.DynamicClient.Resource(gvr).Namespace(namespace)

	// kubectl create -f flunders-1.yaml
	// Request Body: {"apiVersion":"wardle.example.com/v1alpha1","kind":"Flunder","metadata":{"labels":{"sample-label":"true"},"name":"test-flunder","namespace":"default"}}
	testFlunder := samplev1alpha1.Flunder{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Flunder",
			APIVersion: "wardle.example.com/v1alpha1",
		},
		ObjectMeta: metav1.ObjectMeta{Name: flunderName},
		Spec:       samplev1alpha1.FlunderSpec{},
	}
	jsonFlunder, err := json.Marshal(testFlunder)
	framework.ExpectNoError(err, "marshalling test-flunder for create using dynamic client")
	unstruct := &unstructuredv1.Unstructured{}
	err = unstruct.UnmarshalJSON(jsonFlunder)
	framework.ExpectNoError(err, "unmarshalling test-flunder as unstructured for create using dynamic client")
	_, err = dynamicClient.Create(context.TODO(), unstruct, metav1.CreateOptions{})
	framework.ExpectNoError(err, "listing flunders using dynamic client")

	// kubectl get flunders
	unstructuredList, err := dynamicClient.List(context.TODO(), metav1.ListOptions{})
	framework.ExpectNoError(err, "listing flunders using dynamic client")
	if len(unstructuredList.Items) != 1 {
		framework.Failf("failed to get back the correct flunders list %v from the dynamic client", unstructuredList)
	}

	ginkgo.By("Read Status for v1alpha1.wardle.example.com")
	statusContent, err := restClient.Get().
		AbsPath("/apis/apiregistration.k8s.io/v1/apiservices/v1alpha1.wardle.example.com/status").
		SetHeader("Accept", "application/json").DoRaw(context.TODO())
	framework.ExpectNoError(err, "No response for .../apiservices/v1alpha1.wardle.example.com/status. Error: %v", err)

	var jr *apiregistrationv1.APIService
	err = json.Unmarshal([]byte(statusContent), &jr)
	framework.ExpectNoError(err, "Failed to process statusContent: %v | err: %v ", string(statusContent), err)
	framework.ExpectEqual(jr.Status.Conditions[0].Message, "all checks passed", "The Message returned was %v", jr.Status.Conditions[0].Message)

	ginkgo.By("kubectl patch apiservice v1alpha1.wardle.example.com -p '{\"spec\":{\"versionPriority\": 400}}'")
	patchContent, err := restClient.Patch(types.MergePatchType).
		AbsPath("/apis/apiregistration.k8s.io/v1/apiservices/v1alpha1.wardle.example.com").
		SetHeader("Accept", "application/json").
		Body([]byte(`{"spec":{"versionPriority": 400}}`)).DoRaw(context.TODO())

	framework.ExpectNoError(err, "Patch failed for .../apiservices/v1alpha1.wardle.example.com. Error: %v", err)
	err = json.Unmarshal([]byte(patchContent), &jr)
	framework.ExpectNoError(err, "Failed to process patchContent: %v | err: %v ", string(patchContent), err)
	framework.ExpectEqual(jr.Spec.VersionPriority, int32(400), "The VersionPriority returned was %d", jr.Spec.VersionPriority)

	ginkgo.By("List APIServices")
	listApiservices, err := restClient.Get().
		AbsPath("/apis/apiregistration.k8s.io/v1/apiservices").
		SetHeader("Accept", "application/json").DoRaw(context.TODO())

	framework.ExpectNoError(err, "No response for /apis/apiregistration.k8s.io/v1/apiservices Error: %v", err)

	var list *apiregistrationv1.APIServiceList
	err = json.Unmarshal([]byte(listApiservices), &list)
	framework.ExpectNoError(err, "Failed to process APIServiceList: %v | err: %v ", list, err)

	locatedWardle := false
	for _, item := range list.Items {
		if item.Name == "v1alpha1.wardle.example.com" {
			framework.Logf("Found v1alpha1.wardle.example.com in APIServiceList")
			locatedWardle = true
			break
		}
	}
	if !locatedWardle {
		framework.Failf("Unable to find v1alpha1.wardle.example.com in APIServiceList")
	}

	// kubectl delete flunder test-flunder
	err = dynamicClient.Delete(context.TODO(), flunderName, metav1.DeleteOptions{})
	validateErrorWithDebugInfo(f, err, pods, "deleting flunders(%v) using dynamic client", unstructuredList.Items)

	// kubectl get flunders
	unstructuredList, err = dynamicClient.List(context.TODO(), metav1.ListOptions{})
	framework.ExpectNoError(err, "listing flunders using dynamic client")
	if len(unstructuredList.Items) != 0 {
		framework.Failf("failed to get back the correct deleted flunders list %v from the dynamic client", unstructuredList)
	}

	cleanTest(client, aggrclient, namespace)
}

// pollTimed will call Poll but time how long Poll actually took.
// It will then framework.Logf the msg with the duration of the Poll.
// It is assumed that msg will contain one %s for the elapsed time.
func pollTimed(interval, timeout time.Duration, condition wait.ConditionFunc, msg string) error {
	defer func(start time.Time, msg string) {
		elapsed := time.Since(start)
		framework.Logf(msg, elapsed)
	}(time.Now(), msg)
	return wait.Poll(interval, timeout, condition)
}

func validateErrorWithDebugInfo(f *framework.Framework, err error, pods *v1.PodList, msg string, fields ...interface{}) {
	if err != nil {
		namespace := f.Namespace.Name
		msg := fmt.Sprintf(msg, fields...)
		msg += fmt.Sprintf(" but received unexpected error:\n%v", err)
		client := f.ClientSet
		ep, err := client.CoreV1().Endpoints(namespace).Get(context.TODO(), "sample-api", metav1.GetOptions{})
		if err == nil {
			msg += fmt.Sprintf("\nFound endpoints for sample-api:\n%v", ep)
		}
		pds, err := client.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{})
		if err == nil {
			msg += fmt.Sprintf("\nFound pods in %s:\n%v", namespace, pds)
			msg += fmt.Sprintf("\nOriginal pods in %s:\n%v", namespace, pods)
		}

		framework.Failf(msg)
	}
}

func generateFlunderName(base string) string {
	id, err := rand.Int(rand.Reader, big.NewInt(2147483647))
	if err != nil {
		return base
	}
	return fmt.Sprintf("%s-%d", base, id)
}

func checkApiServiceListQuantity(aggrclient *aggregatorclient.Clientset, label string, quantity int) func() (bool, error) {
	return func() (bool, error) {
		var err error

		framework.Logf("Requesting list of APIServices to confirm quantity")

		list, err := aggrclient.ApiregistrationV1().APIServices().List(context.TODO(), metav1.ListOptions{LabelSelector: label})
		if err != nil {
			return false, err
		}

		if len(list.Items) != quantity {
			return false, err
		}
		framework.Logf("Found %d APIService with label %q", quantity, label)
		return true, nil
	}
}

func checkApiServiceStatus(aggrclient *aggregatorclient.Clientset, apiServiceName string, statusConditions apiregistrationv1.APIServiceCondition) func() (bool, error) {
	return func() (bool, error) {

		framework.Logf("Get APIService %q to confirm status", apiServiceName)
		currentApiService, err := aggrclient.ApiregistrationV1().APIServices().Get(context.TODO(), apiServiceName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		for _, cond := range currentApiService.Status.Conditions {
			if cond.Type == statusConditions.Type && cond.Reason == statusConditions.Reason && cond.Message == statusConditions.Message {
				framework.Logf("APIService %q has the required status conditions", apiServiceName)
				return true, nil
			}
		}

		return false, err
	}
}
