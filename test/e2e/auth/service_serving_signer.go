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
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
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

		framework.It("signs Pod certificates and its ClusterTrustBundle can be used to trust them [MinimumKubeletVersion:1.37]", func(ctx context.Context) {
			namespace := f.Namespace.Name

			serverDeploy := serverDeployment(namespace)
			ginkgo.By("Creating server deployment...")
			_, err := f.ClientSet.AppsV1().Deployments(namespace).Create(ctx, serverDeploy, metav1.CreateOptions{})
			if err != nil {
				framework.Failf("failed to create server deployment: %v", err)
			}

			serverSvc := serverService(namespace)
			ginkgo.By("Creating server service...")
			_, err = f.ClientSet.CoreV1().Services(namespace).Create(ctx, serverSvc, metav1.CreateOptions{})
			if err != nil {
				framework.Failf("failed to create server service: %v", err)
			}

			ginkgo.By("Waiting for server deployment to complete...")
			framework.ExpectNoError(testutils.WaitForDeploymentComplete(f.ClientSet, serverDeploy, framework.Logf, time.Second, time.Minute))

			ginkgo.By("Checking server cert is valid for signer's trust bundle")
			serviceAddress := fmt.Sprintf("test-service.%s.svc.%s", namespace, framework.TestContext.ClusterDNSDomain)
			framework.ExpectNoError(checkTLS(ctx, f, serviceAddress, ""), "server cert verification failed")
		})

		framework.It("signs Pod certificate only after the pod-referenced service matches the Pod [MinimumKubeletVersion:1.37]", func(ctx context.Context) {
			namespace := f.Namespace.Name
			serverSvc := serverService(namespace)
			serverSvc.Spec.Selector = map[string]string{
				"something-that": "simply-does-not-match",
			}

			svc, err := f.ClientSet.CoreV1().Services(namespace).Create(ctx, serverSvc, metav1.CreateOptions{})
			framework.ExpectNoError(err, "creating server service")

			serverDeploy := serverDeployment(namespace)
			_, err = f.ClientSet.AppsV1().Deployments(namespace).Create(ctx, serverDeploy, metav1.CreateOptions{})
			framework.ExpectNoError(err, "creating server deployment")

			label := labels.SelectorFromSet(labels.Set(map[string]string{"test-role": "tls-server"}))
			podList, err := e2epod.WaitForPodsWithLabel(ctx, f.ClientSet, namespace, label)
			framework.ExpectNoError(err, "fetching server pod")
			gomega.Expect(podList.Items).To(gomega.HaveLen(1), "unexpected number of server pods")
			gomega.Expect(podList.Items[0].Status.Phase).To(gomega.Equal(v1.PodPending), "the pod should not be able to start without the certificate volume")

			gomega.Eventually(ctx, func(ctx context.Context) bool {
				events, err := f.ClientSet.CoreV1().Events(namespace).List(ctx, metav1.ListOptions{
					FieldSelector: fmt.Sprintf("involvedObject.name=%s,involvedObject.kind=Pod", podList.Items[0].Name),
				})
				framework.ExpectNoError(err, "fetching server pod events")

				foundSignerEvent := false
				for _, e := range events.Items {
					f.Logf("event: %#v", e)
					if e.Type == v1.EventTypeWarning && e.Reason == "ServiceNotMatched" {
						foundSignerEvent = true
						break
					}
				}
				return foundSignerEvent
			}).To(gomega.BeTrueBecause("ServiceNotMatched event is expected"), "looking for events mentioning the service did not match the server pod")

			svc.Spec.Selector = map[string]string{"test-role": "tls-server"}
			_, err = f.ClientSet.CoreV1().Services(namespace).Update(ctx, svc, metav1.UpdateOptions{})
			framework.ExpectNoError(err, "updating the service to match the server pod")

			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, &podList.Items[0]), "the pod recovers when the service starts matching")
		})
		framework.It("signs Pod certificate requesting an FQDN SAN [MinimumKubeletVersion:1.37]", func(ctx context.Context) {
			namespace := f.Namespace.Name

			serverDeploy := serverDeployment(namespace)
			modifyServerToAddFQDN(serverDeploy)
			_, err := f.ClientSet.AppsV1().Deployments(namespace).Create(ctx, serverDeploy, metav1.CreateOptions{})
			framework.ExpectNoError(err, "creating server deployment")

			ginkgo.By("Waiting for server deployment to complete...")
			framework.ExpectNoError(testutils.WaitForDeploymentComplete(f.ClientSet, serverDeploy, framework.Logf, time.Second, time.Minute))

			ginkgo.By("Checking server cert is valid for signer's trust bundle")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"test-role": "tls-server"}))
			podList, err := e2epod.WaitForPodsWithLabel(ctx, f.ClientSet, namespace, label)
			framework.ExpectNoError(err, "fetching server pod")
			gomega.Expect(podList.Items).To(gomega.HaveLen(1), "unexpected number of server pods")

			serverPod := podList.Items[0]
			podAddress := fmt.Sprintf("%s.%s.%s.svc.%s:8443", serverPod.Spec.Hostname, serverPod.Spec.Subdomain, namespace, framework.TestContext.ClusterDNSDomain)
			framework.ExpectNoError(checkTLS(ctx, f, podAddress, serverPod.Status.PodIP), "server cert verification failed")
		})
	})

func checkTLS(ctx context.Context, f *framework.Framework, address string, podIP string) error {
	namespace := f.Namespace.Name
	pod := e2epod.CreateExecPodOrFail(ctx, f.ClientSet, namespace, "tls-client-", func(pod *v1.Pod) {
		pod.Spec.Containers[0].VolumeMounts = []v1.VolumeMount{{
			Name:      "certs",
			ReadOnly:  true,
			MountPath: "/run/tls-config",
		}}
		signerName := serviceServingSignerName
		pod.Spec.Volumes = []v1.Volume{{
			Name: "certs",
			Projected: &v1.ProjectedVolumeSource{
				Sources: []v1.VolumeProjection{{
					ClusterTrustBundle: &v1.ClusterTrustBundleProjection{
						SignerName:    &signerName,
						LabelSelector: &metav1.LabelSelector{},
						Path:          "trust-bundle.pem",
					},
				}},
			},
		}}
	})

	command := []string{
		"curl", "-v", "--silent", "--show-error",
		"--cacert", "/run/tls-config/trust-bundle.pem",
		"https://" + address,
	}

	if len(podIP) != 0 {
		// we'll specify custom resolution to imitate SNI
		command = append(command, "--resolve", fmt.Sprintf("%s:%s", address, podIP))
	}
	_, stderr, err := e2epod.Exec(f.TContext(ctx), e2epod.ExecOptions{
		Command:            command,
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
	container.Ports = []v1.ContainerPort{{
		Name:          "https",
		ContainerPort: 8443,
		Protocol:      v1.ProtocolTCP,
	}}
	container.VolumeMounts = []v1.VolumeMount{{
		Name:      "certs",
		ReadOnly:  true,
		MountPath: "/run/tls-config",
	}}
	d.Spec.Template.Spec.Volumes = []v1.Volume{{
		Name: "certs",
		Projected: &v1.ProjectedVolumeSource{
			Sources: []v1.VolumeProjection{{
				PodCertificate: &v1.PodCertificateProjection{
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

func modifyServerToAddFQDN(d *appsv1.Deployment) {
	d.Spec.Template.Spec.Hostname = "test-hostname"
	d.Spec.Template.Spec.Subdomain = "test-subdomain"
	d.Spec.Template.Spec.Volumes[0].Projected.Sources[0].PodCertificate.UserAnnotations = map[string]string{"kubernetes.io/add-pod-fqdn": "true"}
}

func serverService(namespace string) *v1.Service {
	s := e2eservice.CreateServiceSpec("test-service", "", false, map[string]string{"test-role": "tls-server"})
	s.Namespace = namespace

	s.Spec.Ports = []v1.ServicePort{{
		Name:       "https",
		Port:       443,
		TargetPort: intstr.FromString("https"),
		Protocol:   v1.ProtocolTCP,
	}}

	return s
}
