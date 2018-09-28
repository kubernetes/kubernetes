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
	"crypto/rand"
	"encoding/json"
	"fmt"
	"math/big"
	"strings"
	"time"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	unstructuredv1 "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/client-go/discovery"
	clientset "k8s.io/client-go/kubernetes"
	apiregistrationv1beta1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1"
	aggregatorclient "k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset"
	rbacapi "k8s.io/kubernetes/pkg/apis/rbac"
	utilversion "k8s.io/kubernetes/pkg/util/version"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	samplev1alpha1 "k8s.io/sample-apiserver/pkg/apis/wardle/v1alpha1"

	. "github.com/onsi/ginkgo"
)

var serverAggregatorVersion = utilversion.MustParseSemantic("v1.7.0")

var _ = SIGDescribe("Aggregator", func() {
	var ns string
	var c clientset.Interface
	var aggrclient *aggregatorclient.Clientset

	// BeforeEachs run in LIFO order, AfterEachs run in FIFO order.
	// We want cleanTest to happen before the namespace cleanup AfterEach
	// inserted by NewDefaultFramework, so we put this AfterEach in front
	// of NewDefaultFramework.
	AfterEach(func() {
		cleanTest(c, aggrclient, ns)
	})

	f := framework.NewDefaultFramework("aggregator")

	// We want namespace initialization BeforeEach inserted by
	// NewDefaultFramework to happen before this, so we put this BeforeEach
	// after NewDefaultFramework.
	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		aggrclient = f.AggregatorClient
	})

	It("Should be able to support the 1.7 Sample API Server using the current Aggregator", func() {
		// Make sure the relevant provider supports Agggregator
		framework.SkipUnlessServerVersionGTE(serverAggregatorVersion, f.ClientSet.Discovery())
		framework.SkipUnlessProviderIs("gce", "gke")

		// Testing a 1.7 version of the sample-apiserver
		TestSampleAPIServer(f, imageutils.GetE2EImage(imageutils.APIServer))
	})
})

func cleanTest(client clientset.Interface, aggrclient *aggregatorclient.Clientset, namespace string) {
	// delete the APIService first to avoid causing discovery errors
	_ = aggrclient.ApiregistrationV1beta1().APIServices().Delete("v1alpha1.wardle.k8s.io", nil)

	_ = client.AppsV1().Deployments(namespace).Delete("sample-apiserver", nil)
	_ = client.CoreV1().Secrets(namespace).Delete("sample-apiserver-secret", nil)
	_ = client.CoreV1().Services(namespace).Delete("sample-api", nil)
	_ = client.CoreV1().ServiceAccounts(namespace).Delete("sample-apiserver", nil)
	_ = client.RbacV1beta1().RoleBindings("kube-system").Delete("wardler-auth-reader", nil)
	_ = client.RbacV1beta1().ClusterRoles().Delete("wardler", nil)
	_ = client.RbacV1beta1().ClusterRoleBindings().Delete("wardler:"+namespace+":anonymous", nil)
}

// A basic test if the sample-apiserver code from 1.7 and compiled against 1.7
// will work on the current Aggregator/API-Server.
func TestSampleAPIServer(f *framework.Framework, image string) {
	By("Registering the sample API server.")
	client := f.ClientSet
	restClient := client.Discovery().RESTClient()
	iclient := f.InternalClientset
	aggrclient := f.AggregatorClient

	namespace := f.Namespace.Name
	context := setupServerCert(namespace, "sample-api")
	if framework.ProviderIs("gke") {
		// kubectl create clusterrolebinding user-cluster-admin-binding --clusterrole=cluster-admin --user=user@domain.com
		authenticated := rbacv1beta1.Subject{Kind: rbacv1beta1.GroupKind, Name: user.AllAuthenticated}
		framework.BindClusterRole(client.RbacV1beta1(), "cluster-admin", namespace, authenticated)
	}

	// kubectl create -f namespace.yaml
	// NOTE: aggregated apis should generally be set up in there own namespace. As the test framework is setting up a new namespace, we are just using that.

	// kubectl create -f secret.yaml
	secretName := "sample-apiserver-secret"
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: secretName,
		},
		Type: v1.SecretTypeOpaque,
		Data: map[string][]byte{
			"tls.crt": context.cert,
			"tls.key": context.key,
		},
	}
	_, err := client.CoreV1().Secrets(namespace).Create(secret)
	framework.ExpectNoError(err, "creating secret %q in namespace %q", secretName, namespace)

	// kubectl create -f deploy.yaml
	deploymentName := "sample-apiserver-deployment"
	etcdImage := "quay.io/coreos/etcd:v3.2.24"
	podLabels := map[string]string{"app": "sample-apiserver", "apiserver": "true"}
	replicas := int32(1)
	zero := int64(0)
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
				"--etcd-servers=http://localhost:2379",
				"--tls-cert-file=/apiserver.local.config/certificates/tls.crt",
				"--tls-private-key-file=/apiserver.local.config/certificates/tls.key",
				"--audit-log-path=-",
				"--audit-log-maxage=0",
				"--audit-log-maxbackup=0",
			},
			Image: image,
		},
		{
			Name:  "etcd",
			Image: etcdImage,
		},
	}
	d := &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   deploymentName,
			Labels: podLabels,
		},
		Spec: apps.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: podLabels,
			},
			Strategy: apps.DeploymentStrategy{
				Type: apps.RollingUpdateDeploymentStrategyType,
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
	deployment, err := client.AppsV1().Deployments(namespace).Create(d)
	framework.ExpectNoError(err, "creating deployment %s in namespace %s", deploymentName, namespace)
	err = framework.WaitForDeploymentRevisionAndImage(client, namespace, deploymentName, "1", image)
	framework.ExpectNoError(err, "waiting for the deployment of image %s in %s in %s to complete", image, deploymentName, namespace)
	err = framework.WaitForDeploymentRevisionAndImage(client, namespace, deploymentName, "1", etcdImage)
	framework.ExpectNoError(err, "waiting for the deployment of image %s in %s to complete", etcdImage, deploymentName, namespace)

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
					Port:       443,
					TargetPort: intstr.FromInt(443),
				},
			},
		},
	}
	_, err = client.CoreV1().Services(namespace).Create(service)
	framework.ExpectNoError(err, "creating service %s in namespace %s", "sample-apiserver", namespace)

	// kubectl create -f serviceAccount.yaml
	sa := &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "sample-apiserver"}}
	_, err = client.CoreV1().ServiceAccounts(namespace).Create(sa)
	framework.ExpectNoError(err, "creating service account %s in namespace %s", "sample-apiserver", namespace)

	// kubectl create -f authDelegator.yaml
	_, err = client.RbacV1beta1().ClusterRoleBindings().Create(&rbacv1beta1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "wardler:" + namespace + ":anonymous",
		},
		RoleRef: rbacv1beta1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "wardler",
		},
		Subjects: []rbacv1beta1.Subject{
			{
				APIGroup: "rbac.authorization.k8s.io",
				Kind:     "User",
				Name:     namespace + ":anonymous",
			},
		},
	})
	framework.ExpectNoError(err, "creating cluster role binding %s", "wardler:"+namespace+":anonymous")

	// kubectl create -f role.yaml
	resourceRule, err := rbacapi.NewRule("create", "delete", "deletecollection", "get", "list", "patch", "update", "watch").Groups("wardle.k8s.io").Resources("flunders").Rule()
	framework.ExpectNoError(err, "creating cluster resource rule")
	urlRule, err := rbacapi.NewRule("get").URLs("*").Rule()
	framework.ExpectNoError(err, "creating cluster url rule")
	err = wait.Poll(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		roleLabels := map[string]string{"kubernetes.io/bootstrapping": "wardle-default"}
		role := rbacapi.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "wardler",
				Labels: roleLabels,
			},
			Rules: []rbacapi.PolicyRule{resourceRule, urlRule},
		}
		_, err = iclient.Rbac().ClusterRoles().Create(&role)
		if err != nil {
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err, "creating cluster role wardler - may not have permissions")

	// kubectl create -f auth-reader.yaml
	_, err = client.RbacV1beta1().RoleBindings("kube-system").Create(&rbacv1beta1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "wardler-auth-reader",
			Annotations: map[string]string{
				rbacv1beta1.AutoUpdateAnnotationKey: "true",
			},
		},
		RoleRef: rbacv1beta1.RoleRef{
			APIGroup: "",
			Kind:     "Role",
			Name:     "extension-apiserver-authentication-reader",
		},
		Subjects: []rbacv1beta1.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      "default", // "sample-apiserver",
				Namespace: namespace,
			},
		},
	})
	framework.ExpectNoError(err, "creating role binding %s:sample-apiserver to access configMap", namespace)

	// Wait for the extension apiserver to be up and healthy
	// kubectl get deployments -n <aggregated-api-namespace> && status == Running
	// NOTE: aggregated apis should generally be set up in there own namespace (<aggregated-api-namespace>). As the test framework
	// is setting up a new namespace, we are just using that.
	err = framework.WaitForDeploymentComplete(client, deployment)
	framework.ExpectNoError(err, "deploying extension apiserver in namespace %s", namespace)

	// kubectl create -f apiservice.yaml
	_, err = aggrclient.ApiregistrationV1beta1().APIServices().Create(&apiregistrationv1beta1.APIService{
		ObjectMeta: metav1.ObjectMeta{Name: "v1alpha1.wardle.k8s.io"},
		Spec: apiregistrationv1beta1.APIServiceSpec{
			Service: &apiregistrationv1beta1.ServiceReference{
				Namespace: namespace,
				Name:      "sample-api",
			},
			Group:                "wardle.k8s.io",
			Version:              "v1alpha1",
			CABundle:             context.signingCert,
			GroupPriorityMinimum: 2000,
			VersionPriority:      200,
		},
	})
	framework.ExpectNoError(err, "creating apiservice %s with namespace %s", "v1alpha1.wardle.k8s.io", namespace)

	var (
		currentAPIService *apiregistrationv1beta1.APIService
		currentPods       *v1.PodList
	)

	err = pollTimed(100*time.Millisecond, 60*time.Second, func() (bool, error) {

		currentAPIService, _ = aggrclient.ApiregistrationV1beta1().APIServices().Get("v1alpha1.wardle.k8s.io", metav1.GetOptions{})
		currentPods, _ = client.CoreV1().Pods(namespace).List(metav1.ListOptions{})

		request := restClient.Get().AbsPath("/apis/wardle.k8s.io/v1alpha1/namespaces/default/flunders")
		request.SetHeader("Accept", "application/json")
		_, err := request.DoRaw()
		if err != nil {
			status, ok := err.(*apierrs.StatusError)
			if !ok {
				return false, err
			}
			if status.Status().Code == 503 {
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
					logs, err := framework.GetPodLogs(client, namespace, pod.Name, container.Name)
					framework.Logf("logs of %s/%s (error: %v): %s", pod.Name, container.Name, err, logs)
				}
			}
		}
	}
	framework.ExpectNoError(err, "gave up waiting for apiservice wardle to come up successfully")

	flunderName := generateFlunderName("rest-flunder")

	// kubectl create -f flunders-1.yaml -v 9
	// curl -k -v -XPOST https://localhost/apis/wardle.k8s.io/v1alpha1/namespaces/default/flunders
	// Request Body: {"apiVersion":"wardle.k8s.io/v1alpha1","kind":"Flunder","metadata":{"labels":{"sample-label":"true"},"name":"test-flunder","namespace":"default"}}
	flunder := `{"apiVersion":"wardle.k8s.io/v1alpha1","kind":"Flunder","metadata":{"labels":{"sample-label":"true"},"name":"` + flunderName + `","namespace":"default"}}`
	result := restClient.Post().AbsPath("/apis/wardle.k8s.io/v1alpha1/namespaces/default/flunders").Body([]byte(flunder)).Do()
	framework.ExpectNoError(result.Error(), "creating a new flunders resource")
	var statusCode int
	result.StatusCode(&statusCode)
	if statusCode != 201 {
		framework.Failf("Flunders client creation response was status %d, not 201", statusCode)
	}

	pods, err := client.CoreV1().Pods(namespace).List(metav1.ListOptions{})
	framework.ExpectNoError(result.Error(), "getting pods for flunders service")

	// kubectl get flunders -v 9
	// curl -k -v -XGET https://localhost/apis/wardle.k8s.io/v1alpha1/namespaces/default/flunders
	contents, err := restClient.Get().AbsPath("/apis/wardle.k8s.io/v1alpha1/namespaces/default/flunders").SetHeader("Accept", "application/json").DoRaw()
	framework.ExpectNoError(err, "attempting to get a newly created flunders resource")
	var flundersList samplev1alpha1.FlunderList
	err = json.Unmarshal(contents, &flundersList)
	validateErrorWithDebugInfo(f, err, pods, "Error in unmarshalling %T response from server %s", contents, "/apis/wardle.k8s.io/v1alpha1")
	if len(flundersList.Items) != 1 {
		framework.Failf("failed to get back the correct flunders list %v", flundersList)
	}

	// kubectl delete flunder test-flunder -v 9
	// curl -k -v -XDELETE  https://35.193.112.40/apis/wardle.k8s.io/v1alpha1/namespaces/default/flunders/test-flunder
	_, err = restClient.Delete().AbsPath("/apis/wardle.k8s.io/v1alpha1/namespaces/default/flunders/" + flunderName).DoRaw()
	validateErrorWithDebugInfo(f, err, pods, "attempting to delete a newly created flunders(%v) resource", flundersList.Items)

	// kubectl get flunders -v 9
	// curl -k -v -XGET https://localhost/apis/wardle.k8s.io/v1alpha1/namespaces/default/flunders
	contents, err = restClient.Get().AbsPath("/apis/wardle.k8s.io/v1alpha1/namespaces/default/flunders").SetHeader("Accept", "application/json").DoRaw()
	framework.ExpectNoError(err, "confirming delete of a newly created flunders resource")
	err = json.Unmarshal(contents, &flundersList)
	validateErrorWithDebugInfo(f, err, pods, "Error in unmarshalling %T response from server %s", contents, "/apis/wardle.k8s.io/v1alpha1")
	if len(flundersList.Items) != 0 {
		framework.Failf("failed to get back the correct deleted flunders list %v", flundersList)
	}

	flunderName = generateFlunderName("dynamic-flunder")

	// Rerun the Create/List/Delete tests using the Dynamic client.
	resources, discoveryErr := client.Discovery().ServerPreferredNamespacedResources()
	groupVersionResources, err := discovery.GroupVersionResources(resources)
	framework.ExpectNoError(err, "getting group version resources for dynamic client")
	gvr := schema.GroupVersionResource{Group: "wardle.k8s.io", Version: "v1alpha1", Resource: "flunders"}
	_, ok := groupVersionResources[gvr]
	if !ok {
		framework.Failf("could not find group version resource for dynamic client and wardle/flunders (discovery error: %v, discovery results: %#v)", discoveryErr, groupVersionResources)
	}
	dynamicClient := f.DynamicClient.Resource(gvr).Namespace(namespace)

	// kubectl create -f flunders-1.yaml
	// Request Body: {"apiVersion":"wardle.k8s.io/v1alpha1","kind":"Flunder","metadata":{"labels":{"sample-label":"true"},"name":"test-flunder","namespace":"default"}}
	testFlunder := samplev1alpha1.Flunder{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Flunder",
			APIVersion: "wardle.k8s.io/v1alpha1",
		},
		ObjectMeta: metav1.ObjectMeta{Name: flunderName},
		Spec:       samplev1alpha1.FlunderSpec{},
	}
	jsonFlunder, err := json.Marshal(testFlunder)
	framework.ExpectNoError(err, "marshalling test-flunder for create using dynamic client")
	unstruct := &unstructuredv1.Unstructured{}
	err = unstruct.UnmarshalJSON(jsonFlunder)
	framework.ExpectNoError(err, "unmarshalling test-flunder as unstructured for create using dynamic client")
	unstruct, err = dynamicClient.Create(unstruct, metav1.CreateOptions{})
	framework.ExpectNoError(err, "listing flunders using dynamic client")

	// kubectl get flunders
	unstructuredList, err := dynamicClient.List(metav1.ListOptions{})
	framework.ExpectNoError(err, "listing flunders using dynamic client")
	if len(unstructuredList.Items) != 1 {
		framework.Failf("failed to get back the correct flunders list %v from the dynamic client", unstructuredList)
	}

	// kubectl delete flunder test-flunder
	err = dynamicClient.Delete(flunderName, &metav1.DeleteOptions{})
	validateErrorWithDebugInfo(f, err, pods, "deleting flunders(%v) using dynamic client", unstructuredList.Items)

	// kubectl get flunders
	unstructuredList, err = dynamicClient.List(metav1.ListOptions{})
	framework.ExpectNoError(err, "listing flunders using dynamic client")
	if len(unstructuredList.Items) != 0 {
		framework.Failf("failed to get back the correct deleted flunders list %v from the dynamic client", unstructuredList)
	}

	cleanTest(client, aggrclient, namespace)
}

// pollTimed will call Poll but time how long Poll actually took.
// It will then framework.logf the msg with the duration of the Poll.
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
		ep, err := client.CoreV1().Endpoints(namespace).Get("sample-api", metav1.GetOptions{})
		if err == nil {
			msg += fmt.Sprintf("\nFound endpoints for sample-api:\n%v", ep)
		}
		pds, err := client.CoreV1().Pods(namespace).List(metav1.ListOptions{})
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
