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

package apimachinery

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	"k8s.io/kubernetes/test/utils/crd"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/pointer"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/test/integration"

	// ensure libs have a chance to initialize
	_ "github.com/stretchr/testify/assert"
)

const (
	secretCRDName      = "sample-custom-resource-conversion-webhook-secret"
	deploymentCRDName  = "sample-crd-conversion-webhook-deployment"
	serviceCRDName     = "e2e-test-crd-conversion-webhook"
	roleBindingCRDName = "crd-conversion-webhook-auth-reader"
)

var apiVersions = []apiextensionsv1.CustomResourceDefinitionVersion{
	{
		Name:    "v1",
		Served:  true,
		Storage: true,
		Schema: &apiextensionsv1.CustomResourceValidation{
			OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensionsv1.JSONSchemaProps{
					"hostPort": {Type: "string"},
				},
			},
		},
	},
	{
		Name:    "v2",
		Served:  true,
		Storage: false,
		Schema: &apiextensionsv1.CustomResourceValidation{
			OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensionsv1.JSONSchemaProps{
					"host": {Type: "string"},
					"port": {Type: "string"},
				},
			},
		},
	},
}

var alternativeAPIVersions = []apiextensionsv1.CustomResourceDefinitionVersion{
	{
		Name:    "v1",
		Served:  true,
		Storage: false,
		Schema: &apiextensionsv1.CustomResourceValidation{
			OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensionsv1.JSONSchemaProps{
					"hostPort": {Type: "string"},
				},
			},
		},
	},
	{
		Name:    "v2",
		Served:  true,
		Storage: true,
		Schema: &apiextensionsv1.CustomResourceValidation{
			OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
				Type: "object",
				Properties: map[string]apiextensionsv1.JSONSchemaProps{
					"host": {Type: "string"},
					"port": {Type: "string"},
				},
			},
		},
	},
}

var _ = SIGDescribe("CustomResourceConversionWebhook [Privileged:ClusterAdmin]", func() {
	var certCtx *certContext
	f := framework.NewDefaultFramework("crd-webhook")
	servicePort := int32(9443)
	containerPort := int32(9444)

	var client clientset.Interface
	var namespaceName string

	ginkgo.BeforeEach(func() {
		client = f.ClientSet
		namespaceName = f.Namespace.Name

		ginkgo.By("Setting up server cert")
		certCtx = setupServerCert(f.Namespace.Name, serviceCRDName)
		createAuthReaderRoleBindingForCRDConversion(f, f.Namespace.Name)

		deployCustomResourceWebhookAndService(f, imageutils.GetE2EImage(imageutils.Agnhost), certCtx, servicePort, containerPort)
	})

	ginkgo.AfterEach(func() {
		cleanCRDWebhookTest(client, namespaceName)
	})

	/*
		Release : v1.16
		Testname: Custom Resource Definition Conversion Webhook, conversion custom resource
		Description: Register a conversion webhook and a custom resource definition. Create a v1 custom
		resource. Attempts to read it at v2 MUST succeed.
	*/
	framework.ConformanceIt("should be able to convert from CR v1 to CR v2", func() {
		testcrd, err := crd.CreateMultiVersionTestCRD(f, "stable.example.com", func(crd *apiextensionsv1.CustomResourceDefinition) {
			crd.Spec.Versions = apiVersions
			crd.Spec.Conversion = &apiextensionsv1.CustomResourceConversion{
				Strategy: apiextensionsv1.WebhookConverter,
				Webhook: &apiextensionsv1.WebhookConversion{
					ClientConfig: &apiextensionsv1.WebhookClientConfig{
						CABundle: certCtx.signingCert,
						Service: &apiextensionsv1.ServiceReference{
							Namespace: f.Namespace.Name,
							Name:      serviceCRDName,
							Path:      pointer.StringPtr("/crdconvert"),
							Port:      pointer.Int32Ptr(servicePort),
						},
					},
					ConversionReviewVersions: []string{"v1", "v1beta1"},
				},
			}
			crd.Spec.PreserveUnknownFields = false
		})
		if err != nil {
			return
		}
		defer testcrd.CleanUp()
		waitWebhookConversionReady(f, testcrd.Crd, testcrd.DynamicClients, "v2")
		testCustomResourceConversionWebhook(f, testcrd.Crd, testcrd.DynamicClients)
	})

	/*
		Release : v1.16
		Testname: Custom Resource Definition Conversion Webhook, convert mixed version list
		Description: Register a conversion webhook and a custom resource definition. Create a custom resource stored at
		v1. Change the custom resource definition storage to v2. Create a custom resource stored at v2. Attempt to list
		the custom resources at v2; the list result MUST contain both custom resources at v2.
	*/
	framework.ConformanceIt("should be able to convert a non homogeneous list of CRs", func() {
		testcrd, err := crd.CreateMultiVersionTestCRD(f, "stable.example.com", func(crd *apiextensionsv1.CustomResourceDefinition) {
			crd.Spec.Versions = apiVersions
			crd.Spec.Conversion = &apiextensionsv1.CustomResourceConversion{
				Strategy: apiextensionsv1.WebhookConverter,
				Webhook: &apiextensionsv1.WebhookConversion{
					ClientConfig: &apiextensionsv1.WebhookClientConfig{
						CABundle: certCtx.signingCert,
						Service: &apiextensionsv1.ServiceReference{
							Namespace: f.Namespace.Name,
							Name:      serviceCRDName,
							Path:      pointer.StringPtr("/crdconvert"),
							Port:      pointer.Int32Ptr(servicePort),
						},
					},
					ConversionReviewVersions: []string{"v1", "v1beta1"},
				},
			}
			crd.Spec.PreserveUnknownFields = false
		})
		if err != nil {
			return
		}
		defer testcrd.CleanUp()
		waitWebhookConversionReady(f, testcrd.Crd, testcrd.DynamicClients, "v2")
		testCRListConversion(f, testcrd)
	})
})

func cleanCRDWebhookTest(client clientset.Interface, namespaceName string) {
	_ = client.CoreV1().Services(namespaceName).Delete(context.TODO(), serviceCRDName, metav1.DeleteOptions{})
	_ = client.AppsV1().Deployments(namespaceName).Delete(context.TODO(), deploymentCRDName, metav1.DeleteOptions{})
	_ = client.CoreV1().Secrets(namespaceName).Delete(context.TODO(), secretCRDName, metav1.DeleteOptions{})
	_ = client.RbacV1().RoleBindings("kube-system").Delete(context.TODO(), roleBindingCRDName, metav1.DeleteOptions{})
}

func createAuthReaderRoleBindingForCRDConversion(f *framework.Framework, namespace string) {
	ginkgo.By("Create role binding to let cr conversion webhook read extension-apiserver-authentication")
	client := f.ClientSet
	// Create the role binding to allow the webhook read the extension-apiserver-authentication configmap
	_, err := client.RbacV1().RoleBindings("kube-system").Create(context.TODO(), &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: roleBindingCRDName,
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
		framework.Logf("role binding %s already exists", roleBindingCRDName)
	} else {
		framework.ExpectNoError(err, "creating role binding %s:webhook to access configMap", namespace)
	}
}

func deployCustomResourceWebhookAndService(f *framework.Framework, image string, certCtx *certContext, servicePort int32, containerPort int32) {
	ginkgo.By("Deploying the custom resource conversion webhook pod")
	client := f.ClientSet

	// Creating the secret that contains the webhook's cert.
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: secretCRDName,
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
	podLabels := map[string]string{"app": "sample-crd-conversion-webhook", "crd-webhook": "true"}
	replicas := int32(1)
	zero := int64(0)
	mounts := []v1.VolumeMount{
		{
			Name:      "crd-conversion-webhook-certs",
			ReadOnly:  true,
			MountPath: "/webhook.local.config/certificates",
		},
	}
	volumes := []v1.Volume{
		{
			Name: "crd-conversion-webhook-certs",
			VolumeSource: v1.VolumeSource{
				Secret: &v1.SecretVolumeSource{SecretName: secretCRDName},
			},
		},
	}
	containers := []v1.Container{
		{
			Name:         "sample-crd-conversion-webhook",
			VolumeMounts: mounts,
			Args: []string{
				"crd-conversion-webhook",
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
			Name:   deploymentCRDName,
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
	framework.ExpectNoError(err, "creating deployment %s in namespace %s", deploymentCRDName, namespace)

	ginkgo.By("Wait for the deployment to be ready")

	err = e2edeployment.WaitForDeploymentRevisionAndImage(client, namespace, deploymentCRDName, "1", image)
	framework.ExpectNoError(err, "waiting for the deployment of image %s in %s in %s to complete", image, deploymentCRDName, namespace)

	err = e2edeployment.WaitForDeploymentComplete(client, deployment)
	framework.ExpectNoError(err, "waiting for %s deployment status valid", deploymentCRDName)

	ginkgo.By("Deploying the webhook service")

	serviceLabels := map[string]string{"crd-webhook": "true"}
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      serviceCRDName,
			Labels:    map[string]string{"test": "crd-webhook"},
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
	framework.ExpectNoError(err, "creating service %s in namespace %s", serviceCRDName, namespace)

	ginkgo.By("Verifying the service has paired with the endpoint")
	err = framework.WaitForServiceEndpointsNum(client, namespace, serviceCRDName, 1, 1*time.Second, 30*time.Second)
	framework.ExpectNoError(err, "waiting for service %s/%s have %d endpoint", namespace, serviceCRDName, 1)
}

func verifyV1Object(crd *apiextensionsv1.CustomResourceDefinition, obj *unstructured.Unstructured) {
	gomega.Expect(obj.GetAPIVersion()).To(gomega.BeEquivalentTo(crd.Spec.Group + "/v1"))
	hostPort, exists := obj.Object["hostPort"]
	framework.ExpectEqual(exists, true)

	gomega.Expect(hostPort).To(gomega.BeEquivalentTo("localhost:8080"))
	_, hostExists := obj.Object["host"]
	framework.ExpectEqual(hostExists, false)
	_, portExists := obj.Object["port"]
	framework.ExpectEqual(portExists, false)
}

func verifyV2Object(crd *apiextensionsv1.CustomResourceDefinition, obj *unstructured.Unstructured) {
	gomega.Expect(obj.GetAPIVersion()).To(gomega.BeEquivalentTo(crd.Spec.Group + "/v2"))
	_, hostPortExists := obj.Object["hostPort"]
	framework.ExpectEqual(hostPortExists, false)

	host, hostExists := obj.Object["host"]
	framework.ExpectEqual(hostExists, true)
	gomega.Expect(host).To(gomega.BeEquivalentTo("localhost"))
	port, portExists := obj.Object["port"]
	framework.ExpectEqual(portExists, true)
	gomega.Expect(port).To(gomega.BeEquivalentTo("8080"))
}

func testCustomResourceConversionWebhook(f *framework.Framework, crd *apiextensionsv1.CustomResourceDefinition, customResourceClients map[string]dynamic.ResourceInterface) {
	name := "cr-instance-1"
	ginkgo.By("Creating a v1 custom resource")
	crInstance := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       crd.Spec.Names.Kind,
			"apiVersion": crd.Spec.Group + "/v1",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": f.Namespace.Name,
			},
			"hostPort": "localhost:8080",
		},
	}
	_, err := customResourceClients["v1"].Create(context.TODO(), crInstance, metav1.CreateOptions{})
	gomega.Expect(err).To(gomega.BeNil())
	ginkgo.By("v2 custom resource should be converted")
	v2crd, err := customResourceClients["v2"].Get(context.TODO(), name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Getting v2 of custom resource %s", name)
	verifyV2Object(crd, v2crd)
}

func testCRListConversion(f *framework.Framework, testCrd *crd.TestCrd) {
	crd := testCrd.Crd
	customResourceClients := testCrd.DynamicClients
	name1 := "cr-instance-1"
	name2 := "cr-instance-2"
	ginkgo.By("Creating a v1 custom resource")
	crInstance := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       crd.Spec.Names.Kind,
			"apiVersion": crd.Spec.Group + "/v1",
			"metadata": map[string]interface{}{
				"name":      name1,
				"namespace": f.Namespace.Name,
			},
			"hostPort": "localhost:8080",
		},
	}
	_, err := customResourceClients["v1"].Create(context.TODO(), crInstance, metav1.CreateOptions{})
	gomega.Expect(err).To(gomega.BeNil())

	// Now cr-instance-1 is stored as v1. lets change storage version
	crd, err = integration.UpdateV1CustomResourceDefinitionWithRetry(testCrd.APIExtensionClient, crd.Name, func(c *apiextensionsv1.CustomResourceDefinition) {
		c.Spec.Versions = alternativeAPIVersions
	})
	gomega.Expect(err).To(gomega.BeNil())
	ginkgo.By("Create a v2 custom resource")
	crInstance = &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       crd.Spec.Names.Kind,
			"apiVersion": crd.Spec.Group + "/v1",
			"metadata": map[string]interface{}{
				"name":      name2,
				"namespace": f.Namespace.Name,
			},
			"hostPort": "localhost:8080",
		},
	}

	// After changing a CRD, the resources for versions will be re-created that can be result in
	// cancelled connection (e.g. "grpc connection closed" or "context canceled").
	// Just retrying fixes that.
	//
	// TODO: we have to wait for the storage version to become effective. Storage version changes are not instant.
	for i := 0; i < 5; i++ {
		_, err = customResourceClients["v1"].Create(context.TODO(), crInstance, metav1.CreateOptions{})
		if err == nil {
			break
		}
	}
	gomega.Expect(err).To(gomega.BeNil())

	// Now that we have a v1 and v2 object, both list operation in v1 and v2 should work as expected.

	ginkgo.By("List CRs in v1")
	list, err := customResourceClients["v1"].List(context.TODO(), metav1.ListOptions{})
	gomega.Expect(err).To(gomega.BeNil())
	gomega.Expect(len(list.Items)).To(gomega.BeIdenticalTo(2))
	framework.ExpectEqual((list.Items[0].GetName() == name1 && list.Items[1].GetName() == name2) ||
		(list.Items[0].GetName() == name2 && list.Items[1].GetName() == name1), true)
	verifyV1Object(crd, &list.Items[0])
	verifyV1Object(crd, &list.Items[1])

	ginkgo.By("List CRs in v2")
	list, err = customResourceClients["v2"].List(context.TODO(), metav1.ListOptions{})
	gomega.Expect(err).To(gomega.BeNil())
	gomega.Expect(len(list.Items)).To(gomega.BeIdenticalTo(2))
	framework.ExpectEqual((list.Items[0].GetName() == name1 && list.Items[1].GetName() == name2) ||
		(list.Items[0].GetName() == name2 && list.Items[1].GetName() == name1), true)
	verifyV2Object(crd, &list.Items[0])
	verifyV2Object(crd, &list.Items[1])
}

// waitWebhookConversionReady sends stub custom resource creation requests requiring conversion until one succeeds.
func waitWebhookConversionReady(f *framework.Framework, crd *apiextensionsv1.CustomResourceDefinition, customResourceClients map[string]dynamic.ResourceInterface, version string) {
	framework.ExpectNoError(wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		crInstance := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"kind":       crd.Spec.Names.Kind,
				"apiVersion": crd.Spec.Group + "/" + version,
				"metadata": map[string]interface{}{
					"name":      f.UniqueName,
					"namespace": f.Namespace.Name,
				},
			},
		}
		_, err := customResourceClients[version].Create(context.TODO(), crInstance, metav1.CreateOptions{})
		if err != nil {
			// tolerate clusters that do not set --enable-aggregator-routing and have to wait for kube-proxy
			// to program the service network, during which conversion requests return errors
			framework.Logf("error waiting for conversion to succeed during setup: %v", err)
			return false, nil
		}

		framework.ExpectNoError(customResourceClients[version].Delete(context.TODO(), crInstance.GetName(), metav1.DeleteOptions{}), "cleaning up stub object")
		return true, nil
	}))
}
