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
	"time"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils/crd"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	_ "github.com/stretchr/testify/assert"
)

const (
	secretCRDName      = "sample-custom-resource-conversion-webhook-secret"
	deploymentCRDName  = "sample-crd-conversion-webhook-deployment"
	serviceCRDName     = "e2e-test-crd-conversion-webhook"
	roleBindingCRDName = "crd-conversion-webhook-auth-reader"
)

var serverCRDConversionWebhookVersion = utilversion.MustParseSemantic("v1.13.0-alpha")

var apiVersions = []v1beta1.CustomResourceDefinitionVersion{
	{
		Name:    "v1",
		Served:  true,
		Storage: true,
	},
	{
		Name:    "v2",
		Served:  true,
		Storage: false,
	},
}

var alternativeApiVersions = []v1beta1.CustomResourceDefinitionVersion{
	{
		Name:    "v1",
		Served:  true,
		Storage: false,
	},
	{
		Name:    "v2",
		Served:  true,
		Storage: true,
	},
}

var _ = SIGDescribe("CustomResourceConversionWebhook [Feature:CustomResourceWebhookConversion]", func() {
	var context *certContext
	f := framework.NewDefaultFramework("crd-webhook")

	var client clientset.Interface
	var namespaceName string

	BeforeEach(func() {
		client = f.ClientSet
		namespaceName = f.Namespace.Name

		// Make sure the relevant provider supports conversion webhook
		framework.SkipUnlessServerVersionGTE(serverCRDConversionWebhookVersion, f.ClientSet.Discovery())

		By("Setting up server cert")
		context = setupServerCert(f.Namespace.Name, serviceCRDName)
		createAuthReaderRoleBindingForCRDConversion(f, f.Namespace.Name)

		deployCustomResourceWebhookAndService(f, imageutils.GetE2EImage(imageutils.CRDConversionWebhook), context)
	})

	AfterEach(func() {
		cleanCRDWebhookTest(client, namespaceName)
	})

	It("Should be able to convert from CR v1 to CR v2", func() {
		testcrd, err := crd.CreateMultiVersionTestCRD(f, "stable.example.com", apiVersions,
			&v1beta1.WebhookClientConfig{
				CABundle: context.signingCert,
				Service: &v1beta1.ServiceReference{
					Namespace: f.Namespace.Name,
					Name:      serviceCRDName,
					Path:      strPtr("/crdconvert"),
				}})
		if err != nil {
			return
		}
		defer testcrd.CleanUp()
		testCustomResourceConversionWebhook(f, testcrd.Crd, testcrd.DynamicClients)
	})

	It("Should be able to convert a non homogeneous list of CRs", func() {
		testcrd, err := crd.CreateMultiVersionTestCRD(f, "stable.example.com", apiVersions,
			&v1beta1.WebhookClientConfig{
				CABundle: context.signingCert,
				Service: &v1beta1.ServiceReference{
					Namespace: f.Namespace.Name,
					Name:      serviceCRDName,
					Path:      strPtr("/crdconvert"),
				}})
		if err != nil {
			return
		}
		defer testcrd.CleanUp()
		testCRListConversion(f, testcrd)
	})
})

func cleanCRDWebhookTest(client clientset.Interface, namespaceName string) {
	_ = client.CoreV1().Services(namespaceName).Delete(serviceCRDName, nil)
	_ = client.AppsV1().Deployments(namespaceName).Delete(deploymentCRDName, nil)
	_ = client.CoreV1().Secrets(namespaceName).Delete(secretCRDName, nil)
	_ = client.RbacV1().RoleBindings("kube-system").Delete(roleBindingCRDName, nil)
}

func createAuthReaderRoleBindingForCRDConversion(f *framework.Framework, namespace string) {
	By("Create role binding to let cr conversion webhook read extension-apiserver-authentication")
	client := f.ClientSet
	// Create the role binding to allow the webhook read the extension-apiserver-authentication configmap
	_, err := client.RbacV1().RoleBindings("kube-system").Create(&rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: roleBindingCRDName,
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "",
			Kind:     "Role",
			Name:     "extension-apiserver-authentication-reader",
		},
		// Webhook uses the default service account.
		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      "default",
				Namespace: namespace,
			},
		},
	})
	if err != nil && errors.IsAlreadyExists(err) {
		framework.Logf("role binding %s already exists", roleBindingCRDName)
	} else {
		framework.ExpectNoError(err, "creating role binding %s:webhook to access configMap", namespace)
	}
}

func deployCustomResourceWebhookAndService(f *framework.Framework, image string, context *certContext) {
	By("Deploying the custom resource conversion webhook pod")
	client := f.ClientSet

	// Creating the secret that contains the webhook's cert.
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: secretCRDName,
		},
		Type: v1.SecretTypeOpaque,
		Data: map[string][]byte{
			"tls.crt": context.cert,
			"tls.key": context.key,
		},
	}
	namespace := f.Namespace.Name
	_, err := client.CoreV1().Secrets(namespace).Create(secret)
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
				"--tls-cert-file=/webhook.local.config/certificates/tls.crt",
				"--tls-private-key-file=/webhook.local.config/certificates/tls.key",
				"--alsologtostderr",
				"-v=4",
				"2>&1",
			},
			Image: image,
		},
	}
	d := &apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   deploymentCRDName,
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
	framework.ExpectNoError(err, "creating deployment %s in namespace %s", deploymentCRDName, namespace)
	By("Wait for the deployment to be ready")
	err = framework.WaitForDeploymentRevisionAndImage(client, namespace, deploymentCRDName, "1", image)
	framework.ExpectNoError(err, "waiting for the deployment of image %s in %s in %s to complete", image, deploymentName, namespace)
	err = framework.WaitForDeploymentComplete(client, deployment)
	framework.ExpectNoError(err, "waiting for the deployment status valid", image, deploymentCRDName, namespace)

	By("Deploying the webhook service")

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
					Port:       443,
					TargetPort: intstr.FromInt(443),
				},
			},
		},
	}
	_, err = client.CoreV1().Services(namespace).Create(service)
	framework.ExpectNoError(err, "creating service %s in namespace %s", serviceCRDName, namespace)

	By("Verifying the service has paired with the endpoint")
	err = framework.WaitForServiceEndpointsNum(client, namespace, serviceCRDName, 1, 1*time.Second, 30*time.Second)
	framework.ExpectNoError(err, "waiting for service %s/%s have %d endpoint", namespace, serviceCRDName, 1)
}

func verifyV1Object(f *framework.Framework, crd *v1beta1.CustomResourceDefinition, obj *unstructured.Unstructured) {
	Expect(obj.GetAPIVersion()).To(BeEquivalentTo(crd.Spec.Group + "/v1"))
	hostPort, exists := obj.Object["hostPort"]
	Expect(exists).To(BeTrue())
	Expect(hostPort).To(BeEquivalentTo("localhost:8080"))
	_, hostExists := obj.Object["host"]
	Expect(hostExists).To(BeFalse())
	_, portExists := obj.Object["port"]
	Expect(portExists).To(BeFalse())
}

func verifyV2Object(f *framework.Framework, crd *v1beta1.CustomResourceDefinition, obj *unstructured.Unstructured) {
	Expect(obj.GetAPIVersion()).To(BeEquivalentTo(crd.Spec.Group + "/v2"))
	_, hostPortExists := obj.Object["hostPort"]
	Expect(hostPortExists).To(BeFalse())
	host, hostExists := obj.Object["host"]
	Expect(hostExists).To(BeTrue())
	Expect(host).To(BeEquivalentTo("localhost"))
	port, portExists := obj.Object["port"]
	Expect(portExists).To(BeTrue())
	Expect(port).To(BeEquivalentTo("8080"))
}

func testCustomResourceConversionWebhook(f *framework.Framework, crd *v1beta1.CustomResourceDefinition, customResourceClients map[string]dynamic.ResourceInterface) {
	name := "cr-instance-1"
	By("Creating a v1 custom resource")
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
	_, err := customResourceClients["v1"].Create(crInstance, metav1.CreateOptions{})
	Expect(err).To(BeNil())
	By("v2 custom resource should be converted")
	v2crd, err := customResourceClients["v2"].Get(name, metav1.GetOptions{})
	verifyV2Object(f, crd, v2crd)
}

func testCRListConversion(f *framework.Framework, testCrd *crd.TestCrd) {
	crd := testCrd.Crd
	customResourceClients := testCrd.DynamicClients
	name1 := "cr-instance-1"
	name2 := "cr-instance-2"
	By("Creating a v1 custom resource")
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
	_, err := customResourceClients["v1"].Create(crInstance, metav1.CreateOptions{})
	Expect(err).To(BeNil())

	// Now cr-instance-1 is stored as v1. lets change storage version
	crd, err = integration.UpdateCustomResourceDefinitionWithRetry(testCrd.APIExtensionClient, crd.Name, func(c *v1beta1.CustomResourceDefinition) {
		c.Spec.Versions = alternativeApiVersions
	})
	Expect(err).To(BeNil())
	By("Create a v2 custom resource")
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
	for i := 0; i < 5; i++ {
		_, err = customResourceClients["v1"].Create(crInstance, metav1.CreateOptions{})
		if err == nil {
			break
		}
	}
	Expect(err).To(BeNil())

	// Now that we have a v1 and v2 object, both list operation in v1 and v2 should work as expected.

	By("List CRs in v1")
	list, err := customResourceClients["v1"].List(metav1.ListOptions{})
	Expect(err).To(BeNil())
	Expect(len(list.Items)).To(BeIdenticalTo(2))
	Expect((list.Items[0].GetName() == name1 && list.Items[1].GetName() == name2) ||
		(list.Items[0].GetName() == name2 && list.Items[1].GetName() == name1)).To(BeTrue())
	verifyV1Object(f, crd, &list.Items[0])
	verifyV1Object(f, crd, &list.Items[1])

	By("List CRs in v2")
	list, err = customResourceClients["v2"].List(metav1.ListOptions{})
	Expect(err).To(BeNil())
	Expect(len(list.Items)).To(BeIdenticalTo(2))
	Expect((list.Items[0].GetName() == name1 && list.Items[1].GetName() == name2) ||
		(list.Items[0].GetName() == name2 && list.Items[1].GetName() == name1)).To(BeTrue())
	verifyV2Object(f, crd, &list.Items[0])
	verifyV2Object(f, crd, &list.Items[1])
}
