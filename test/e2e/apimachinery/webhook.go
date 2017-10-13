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
	"strings"
	"time"

	"k8s.io/api/admissionregistration/v1alpha1"
	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilversion "k8s.io/kubernetes/pkg/util/version"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	_ "github.com/stretchr/testify/assert"
)

const (
	secretName        = "sample-webhook-secret"
	deploymentName    = "sample-webhook-deployment"
	serviceName       = "e2e-test-webhook"
	roleBindingName   = "webhook-auth-reader"
	webhookConfigName = "e2e-test-webhook-config"
)

var serverWebhookVersion = utilversion.MustParseSemantic("v1.8.0")

var _ = SIGDescribe("AdmissionWebhook", func() {
	f := framework.NewDefaultFramework("webhook")
	framework.AddCleanupAction(func() {
		cleanWebhookTest(f)
	})

	It("Should be able to deny pod creation", func() {
		// Make sure the relevant provider supports admission webhook
		framework.SkipUnlessServerVersionGTE(serverWebhookVersion, f.ClientSet.Discovery())
		framework.SkipUnlessProviderIs("gce", "gke")

		_, err := f.ClientSet.AdmissionregistrationV1alpha1().ExternalAdmissionHookConfigurations().List(metav1.ListOptions{})
		if errors.IsNotFound(err) {
			framework.Skipf("dynamic configuration of webhooks requires the alpha admissionregistration.k8s.io group to be enabled")
		}

		By("Setting up server cert")
		namespaceName := f.Namespace.Name
		context := setupServerCert(namespaceName, serviceName)
		createAuthReaderRoleBinding(f, namespaceName)
		// Note that in 1.9 we will have backwards incompatible change to
		// admission webhooks, so the image will be updated to 1.9 sometime in
		// the development 1.9 cycle.
		deployWebhookAndService(f, "gcr.io/kubernetes-e2e-test-images/k8s-sample-admission-webhook-amd64:1.8v1", context)
		registerWebhook(f, context)
		testWebhook(f)
	})
})

func createAuthReaderRoleBinding(f *framework.Framework, namespace string) {
	By("Create role binding to let webhook read extension-apiserver-authentication")
	client := f.ClientSet
	// Create the role binding to allow the webhook read the extension-apiserver-authentication configmap
	_, err := client.RbacV1beta1().RoleBindings("kube-system").Create(&rbacv1beta1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: roleBindingName,
			Annotations: map[string]string{
				rbacv1beta1.AutoUpdateAnnotationKey: "true",
			},
		},
		RoleRef: rbacv1beta1.RoleRef{
			APIGroup: "",
			Kind:     "Role",
			Name:     "extension-apiserver-authentication-reader",
		},
		// Webhook uses the default service account.
		Subjects: []rbacv1beta1.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      "default",
				Namespace: namespace,
			},
		},
	})
	framework.ExpectNoError(err, "creating role binding %s:webhook to access configMap", namespace)
}

func deployWebhookAndService(f *framework.Framework, image string, context *certContext) {
	By("Deploying the webhook pod")

	client := f.ClientSet

	// Creating the secret that contains the webhook's cert.
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
	namespace := f.Namespace.Name
	_, err := client.CoreV1().Secrets(namespace).Create(secret)
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
				"--tls-cert-file=/webhook.local.config/certificates/tls.crt",
				"--tls-private-key-file=/webhook.local.config/certificates/tls.key",
				"--alsologtostderr",
				"-v=4",
				"2>&1",
			},
			Image: image,
		},
	}
	d := &extensions.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: deploymentName,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: &replicas,
			Strategy: extensions.DeploymentStrategy{
				Type: extensions.RollingUpdateDeploymentStrategyType,
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
	deployment, err := client.ExtensionsV1beta1().Deployments(namespace).Create(d)
	framework.ExpectNoError(err, "creating deployment %s in namespace %s", deploymentName, namespace)
	By("Wait for the deployment to be ready")
	err = framework.WaitForDeploymentRevisionAndImage(client, namespace, deploymentName, "1", image)
	framework.ExpectNoError(err, "waiting for the deployment of image %s in %s in %s to complete", image, deploymentName, namespace)
	err = framework.WaitForDeploymentComplete(client, deployment)
	framework.ExpectNoError(err, "waiting for the deployment status valid", image, deploymentName, namespace)

	By("Deploying the webhook service")

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
					Port:       443,
					TargetPort: intstr.FromInt(443),
				},
			},
		},
	}
	_, err = client.CoreV1().Services(namespace).Create(service)
	framework.ExpectNoError(err, "creating service %s in namespace %s", serviceName, namespace)

	By("Verifying the service has paired with the endpoint")
	err = framework.WaitForServiceEndpointsNum(client, namespace, serviceName, 1, 1*time.Second, 30*time.Second)
	framework.ExpectNoError(err, "waiting for service %s/%s have %d endpoint", namespace, serviceName, 1)
}

func registerWebhook(f *framework.Framework, context *certContext) {
	client := f.ClientSet
	By("Registering the webhook via the AdmissionRegistration API")

	namespace := f.Namespace.Name
	_, err := client.AdmissionregistrationV1alpha1().ExternalAdmissionHookConfigurations().Create(&v1alpha1.ExternalAdmissionHookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: webhookConfigName,
		},
		ExternalAdmissionHooks: []v1alpha1.ExternalAdmissionHook{
			{
				Name: "e2e-test-webhook.k8s.io",
				Rules: []v1alpha1.RuleWithOperations{{
					Operations: []v1alpha1.OperationType{v1alpha1.Create},
					Rule: v1alpha1.Rule{
						APIGroups:   []string{""},
						APIVersions: []string{"v1"},
						Resources:   []string{"pods"},
					},
				}},
				ClientConfig: v1alpha1.AdmissionHookClientConfig{
					Service: v1alpha1.ServiceReference{
						Namespace: namespace,
						Name:      serviceName,
					},
					CABundle: context.signingCert,
				},
			},
		},
	})
	framework.ExpectNoError(err, "registering webhook config %s with namespace %s", webhookConfigName, namespace)

	// The webhook configuration is honored in 1s.
	time.Sleep(2 * time.Second)
}

func testWebhook(f *framework.Framework) {
	By("create a pod that should be denied by the webhook")
	client := f.ClientSet
	// Creating the pod, the request should be rejected
	pod := nonCompliantPod(f)
	_, err := client.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).NotTo(BeNil())
	expectedErrMsg := "the pod contains unwanted container name"
	if !strings.Contains(err.Error(), expectedErrMsg) {
		framework.Failf("expect error contains %q, got %q", expectedErrMsg, err.Error())
	}
	// TODO: Test if webhook can detect pod with non-compliant metadata.
	// Currently metadata is lost because webhook uses the external version of
	// the objects, and the apiserver sends the internal objects.
}

func nonCompliantPod(f *framework.Framework) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "disallowed-pod",
			Labels: map[string]string{
				"webhook-e2e-test": "disallow",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "webhook-disallow",
					Image: framework.GetPauseImageName(f.ClientSet),
				},
			},
		},
	}
}

func cleanWebhookTest(f *framework.Framework) {
	client := f.ClientSet
	_ = client.AdmissionregistrationV1alpha1().ExternalAdmissionHookConfigurations().Delete(webhookConfigName, nil)
	namespaceName := f.Namespace.Name
	_ = client.CoreV1().Services(namespaceName).Delete(serviceName, nil)
	_ = client.ExtensionsV1beta1().Deployments(namespaceName).Delete(deploymentName, nil)
	_ = client.CoreV1().Secrets(namespaceName).Delete(secretName, nil)
	_ = client.RbacV1beta1().RoleBindings("kube-system").Delete(roleBindingName, nil)
}
