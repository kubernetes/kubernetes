/*
Copyright The Kubernetes Authors.

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
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/features"

	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

const serviceServingSignerName = "kubernetes.io/service-serving"

var _ = SIGDescribe("Service Serving Signer",
	framework.WithFeatureGate(features.PodCertificateRequest),
	framework.WithFeatureGate(features.ClusterTrustBundle),
	framework.WithFeatureGate(features.ClusterTrustBundleProjection),
	framework.WithFeatureGate(features.KubeServingCertificatesSigner),
	func() {
		f := framework.NewDefaultFramework("serviceservingsigner")
		f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

		framework.It("kubernetes.io/service-serving signs Pod certificates and its ClusterTrustBundle can be used to trust them [MinimumKubeletVersion:1.37]", func(ctx context.Context) {
			namespace := f.Namespace.Name

			serverDeploy := serverDeployment(namespace)
			ginkgo.By("Creating server deployment...")
			_, err := f.ClientSet.AppsV1().Deployments(namespace).Create(ctx, serverDeploy, metav1.CreateOptions{})
			if err != nil {
				framework.Failf("failed to create server deployment: %v", err)
			}

			serverSvc := serverService()
			ginkgo.By("Creating server service...")
			_, err = f.ClientSet.CoreV1().Services(namespace).Create(ctx, serverSvc, metav1.CreateOptions{})
			if err != nil {
				framework.Failf("failed to create server service: %v", err)
			}

			ginkgo.By("Waiting for server deployment to complete...")
			framework.ExpectNoError(testutils.WaitForDeploymentComplete(f.ClientSet, serverDeploy, framework.Logf, time.Second, time.Minute))

			ginkgo.By("Checking server cert is valid for signer's trust bundle")
			framework.ExpectNoError(checkServiceServingTLS(ctx, f), "server cert verification failed")
		})
	})

func checkServiceServingTLS(ctx context.Context, f *framework.Framework) error {
	namespace := f.Namespace.Name
	pod := e2epod.CreateExecPodOrFail(ctx, f.ClientSet, namespace, "tls-client-", func(pod *corev1.Pod) {
		pod.Spec.Containers[0].VolumeMounts = []corev1.VolumeMount{{
			Name:      "certs",
			ReadOnly:  true,
			MountPath: "/run/tls-config",
		}}
		signerName := serviceServingSignerName
		pod.Spec.Volumes = []corev1.Volume{{
			Name: "certs",
			Projected: &corev1.ProjectedVolumeSource{
				Sources: []corev1.VolumeProjection{{
					ClusterTrustBundle: &corev1.ClusterTrustBundleProjection{
						SignerName:    &signerName,
						LabelSelector: &metav1.LabelSelector{},
						Path:          "trust-bundle.pem",
					},
				}},
			},
		}}
	})
	_, stderr, err := e2epod.Exec(f.TContext(ctx), e2epod.ExecOptions{
		Command: []string{
			"curl", "-v", "--silent", "--show-error", "--max-time", "10",
			"--retry", "5", "--retry-delay", "2", "--retry-all-errors", "--retry-max-time", "30",
			"--cacert", "/run/tls-config/trust-bundle.pem",
			fmt.Sprintf("https://test-service.%s.svc.%s", namespace, framework.TestContext.ClusterDNSDomain),
		},
		Namespace:          namespace,
		PodName:            pod.Name,
		ContainerName:      pod.Spec.Containers[0].Name,
		CaptureStderr:      true,
		PreserveWhitespace: true,
	})
	framework.Logf("\ncurl stderr:\n%s", stderr)
	if err != nil {
		return fmt.Errorf("curl failed to verify the server TLS connection: %w", err)
	}
	return nil
}

func serverDeployment(namespace string) *appsv1.Deployment {
	d := e2edeployment.NewDeployment("test-server", int32(1), map[string]string{"test-role": "tls-server"}, "tls-server", imageutils.GetE2EImage(imageutils.Agnhost), appsv1.RollingUpdateDeploymentStrategyType)
	d.Namespace = namespace

	container := &d.Spec.Template.Spec.Containers[0]
	container.Args = []string{"webhook", "--port=8443", "--tls-cert-file=/run/tls-config/tls.crt", "--tls-private-key-file=/run/tls-config/tls.key"}
	container.Ports = []corev1.ContainerPort{{
		Name:          "https",
		ContainerPort: 8443,
		Protocol:      corev1.ProtocolTCP,
	}}
	container.VolumeMounts = []corev1.VolumeMount{{
		Name:      "certs",
		ReadOnly:  true,
		MountPath: "/run/tls-config",
	}}
	d.Spec.Template.Spec.Volumes = []corev1.Volume{{
		Name: "certs",
		Projected: &corev1.ProjectedVolumeSource{
			Sources: []corev1.VolumeProjection{{
				PodCertificate: &corev1.PodCertificateProjection{
					SignerName:           serviceServingSignerName,
					KeyType:              "RSA4096",
					CertificateChainPath: "tls.crt",
					KeyPath:              "tls.key",
					UserAnnotations: map[string]string{
						"kubernetes.io/service-names": "test-service",
					},
				},
			}},
		},
	}}

	return d
}

func serverService() *corev1.Service {
	s := e2eservice.CreateServiceSpec("test-service", "", false, map[string]string{"test-role": "tls-server"})
	s.Spec.Ports = []corev1.ServicePort{{
		Name:       "https",
		Port:       443,
		TargetPort: intstr.FromString("https"),
		Protocol:   corev1.ProtocolTCP,
	}}

	return s
}
