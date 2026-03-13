/*
Copyright 2024 The Kubernetes Authors.

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
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/hermeticpodcertificatesigner"
	imageutils "k8s.io/kubernetes/test/utils/image" // Import imageutils
	admissionapi "k8s.io/pod-security-admission/api"
	admissiontest "k8s.io/pod-security-admission/test"
	"k8s.io/utils/clock"
)

var _ = SIGDescribe("Projected PodCertificate",
	framework.WithFeatureGate(features.PodCertificateRequest),
	framework.WithFeatureGate(features.ClusterTrustBundle),
	framework.WithFeatureGate(features.ClusterTrustBundleProjection),
	func() {
		f := framework.NewDefaultFramework("projected-podcertificate")
		f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

		var spiffeSignerName string

		ginkgo.BeforeEach(func(ctx context.Context) {
			spiffeSignerName = "e2e.example.com/" + f.UniqueName

			ginkgo.By("Starting in-process pod certificate signer...")
			signerCtx, cancelSigner := context.WithCancel(context.Background())
			ginkgo.DeferCleanup(func(ctx context.Context) {
				ginkgo.By("Stopping in-process pod certificate signer...")
				cancelSigner()
			})

			var err error
			caKeys, caCerts, err := hermeticpodcertificatesigner.GenerateCAHierarchy(1) // Generate CA once
			if err != nil {
				framework.Failf("failed to generate CA for signer: %v", err)
			}

			signer := hermeticpodcertificatesigner.New(clock.RealClock{}, spiffeSignerName, caKeys, caCerts, f.ClientSet)
			go signer.Run(signerCtx)
		})

		ginkgo.It("should allow server and client pods to establish an mTLS connection", func(ctx context.Context) {
			namespace := f.Namespace.Name
			ginkgo.By("Using namespace: " + namespace)

			securityContext := generateContainerSecurityContext()
			serverDeployment, serverService := createServerObjects(namespace, spiffeSignerName, securityContext)
			ginkgo.By("Creating server deployment...")
			_, err := f.ClientSet.AppsV1().Deployments(namespace).Create(ctx, serverDeployment, metav1.CreateOptions{})
			if err != nil {
				framework.Failf("failed to create server deployment: %v", err)
			}

			ginkgo.By("Creating server service...")
			_, err = f.ClientSet.CoreV1().Services(namespace).Create(ctx, serverService, metav1.CreateOptions{})
			if err != nil {
				framework.Failf("failed to create server service: %v", err)
			}

			clientDeployment := createClientObjects(namespace, spiffeSignerName, securityContext)
			ginkgo.By("Creating client deployment...")
			_, err = f.ClientSet.AppsV1().Deployments(namespace).Create(ctx, clientDeployment, metav1.CreateOptions{})
			if err != nil {
				framework.Failf("failed to create client deployment: %v", err)
			}

			ginkgo.By("Waiting for mTLS connection to be built...")

			// The mtlsclient now logs successful polls, check for that pattern.
			gomega.Eventually(ctx, func(ctx context.Context) (string, error) {
				clientLabels := labels.Set{"app": "client"}
				podList, listErr := f.ClientSet.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: clientLabels.String()})
				if listErr != nil || len(podList.Items) == 0 {
					return "", fmt.Errorf("failed to get client pods: %w", err)
				}
				return e2epod.GetPodLogs(ctx, f.ClientSet, namespace, podList.Items[0].Name, "client")
			}, 30*time.Second, 2*time.Second).Should(gomega.MatchRegexp(`Got response body: Client Identity: spiffe://`),
				"client logs did not contain expected success message pattern")
		})

		ginkgo.It("should honor UserAnnotations for SPIFFE URI path", func(ctx context.Context) {
			namespace := f.Namespace.Name
			ginkgo.By("Using namespace: " + namespace)

			// Use customerPath to override the path in the spiffeID.
			customPath := "/custom-workload"
			userAnnotations := map[string]string{
				"spiffe/path-overriding": customPath, // Match the key supported the signer
			}
			inspectorPod := createInspectorPod(namespace, "path-override-pod", userAnnotations, nil, spiffeSignerName)

			ginkgo.By("Creating inspector pod with UserAnnotations...")
			_, err := f.ClientSet.CoreV1().Pods(namespace).Create(ctx, inspectorPod, metav1.CreateOptions{})
			if err != nil {
				framework.Failf("failed to create inspector pod: %v", err)
			}

			ginkgo.By("Waiting for inspector pod to be running...")
			err = e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, inspectorPod)
			if err != nil {
				framework.Failf("inspector pod failed to start: %v", err)
			}

			ginkgo.By("Fetching and parsing certificate from pod logs...")
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, namespace, inspectorPod.Name, inspectorPod.Spec.Containers[0].Name)
			if err != nil {
				framework.Failf("failed to get pod logs: %v", err)
			}
			certs, err := certutil.ParseCertsPEM([]byte(logs))
			if err != nil {
				framework.Failf("failed to parse certificate chain from PEM block: %v", err)
			}
			gomega.Expect(certs).ToNot(gomega.BeEmpty(), "should parse at least one certificate from the bundle")

			cert := certs[0]
			foundSPIFFEURI := false
			for _, uri := range cert.URIs {
				if uri.Scheme == "spiffe" {
					foundSPIFFEURI = true
					ginkgo.By("Found SPIFFE URI: " + uri.String())
					gomega.Expect(uri.Path).To(gomega.Equal(customPath), "SPIFFE URI path should match the custom path from UserAnnotations")
					break
				}
			}
			gomega.Expect(foundSPIFFEURI).To(gomega.BeTrueBecause("should find a SPIFFE URI in the certificate's SANs"))
		})

		ginkgo.Describe("MaxExpirationSeconds validations", func() {

			ginkgo.It("should issue certificate with default life time (24h) when MaxExpirationSeconds is not set", func(ctx context.Context) {
				namespace := f.Namespace.Name
				ginkgo.By("Using namespace: " + namespace)

				// Create pod without MaxExpirationSeconds
				testPod := createInspectorPod(namespace, "default-duration-pod", nil, nil, spiffeSignerName)
				ginkgo.By("Creating pod without MaxExpirationSeconds...")
				createdPod, err := f.ClientSet.CoreV1().Pods(namespace).Create(ctx, testPod, metav1.CreateOptions{})
				if err != nil {
					framework.Failf("failed to create pod: %v", err)
				}

				ginkgo.By("Waiting for pod to be running...")
				err = e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, createdPod)
				if err != nil {
					framework.Failf("Pod failed to start: %v", err)
				}

				ginkgo.By("Fetching certificate and verifying duration...")
				logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, namespace, createdPod.Name, createdPod.Spec.Containers[0].Name)
				if err != nil {
					framework.Failf("failed to get pod logs: %v", err)
				}
				certs, err := certutil.ParseCertsPEM([]byte(logs))
				if err != nil {
					framework.Failf("failed to parse certificate: %v", err)
				}
				gomega.Expect(certs).ToNot(gomega.BeEmpty())
				cert := certs[0]
				lifeTime := cert.NotAfter.Sub(cert.NotBefore)
				expectedDuration := 24 * time.Hour // Default from signer code
				ginkgo.By(fmt.Sprintf("Verifying certificate duration %v is close to %v", lifeTime, expectedDuration))
				gomega.Expect(lifeTime).To(gomega.BeNumerically("==", expectedDuration), "Certificate duration should be 24 hours")
			})

			ginkgo.It("should issue certificate with specified duration (1h) when MaxExpirationSeconds is set", func(ctx context.Context) {
				namespace := f.Namespace.Name
				ginkgo.By("Using namespace: " + namespace)

				// Create pod requesting 1 hour
				requestedSeconds := int32(3600)
				testPod := createInspectorPod(namespace, "one-hour-duration-pod", nil, &requestedSeconds, spiffeSignerName)
				ginkgo.By("Creating pod requesting 1 hour MaxExpirationSeconds...")
				createdPod, err := f.ClientSet.CoreV1().Pods(namespace).Create(ctx, testPod, metav1.CreateOptions{})
				if err != nil {
					framework.Failf("failed to create pod: %v", err)
				}

				ginkgo.By("Waiting for pod to be running...")
				err = e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, createdPod)
				if err != nil {
					framework.Failf("Pod failed to start: %v", err)
				}

				ginkgo.By("Fetching certificate and verifying duration...")
				logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, namespace, createdPod.Name, createdPod.Spec.Containers[0].Name)
				if err != nil {
					framework.Failf("failed to get pod logs: %v", err)
				}
				certs, err := certutil.ParseCertsPEM([]byte(logs))
				if err != nil {
					framework.Failf("failed to parse certificate: %v", err)
				}
				gomega.Expect(certs).ToNot(gomega.BeEmpty())
				cert := certs[0]
				lifeTime := cert.NotAfter.Sub(cert.NotBefore)
				expectedDuration := time.Duration(requestedSeconds) * time.Second
				ginkgo.By(fmt.Sprintf("Verifying certificate duration %v is close to %v", lifeTime, expectedDuration))
				gomega.Expect(lifeTime).To(gomega.BeNumerically("==", expectedDuration), "Certificate duration should be 1 hour")
			})

			ginkgo.It("should fail pod startup when MaxExpirationSeconds exceeds maximum (91d)", func(ctx context.Context) {
				namespace := f.Namespace.Name
				ginkgo.By("Using namespace: " + namespace)

				// Exceeds 91 days
				tooLongSeconds := int32((91 * 24 * 60 * 60) + 1)
				testPod := createInspectorPod(namespace, "too-long-duration-pod", nil, &tooLongSeconds, spiffeSignerName)
				ginkgo.By("Creating pod requesting >91d MaxExpirationSeconds...")
				_, err := f.ClientSet.CoreV1().Pods(namespace).Create(ctx, testPod, metav1.CreateOptions{})
				if err == nil {
					framework.Fail("Error message does not match the expected.")
				}
				if err != nil {
					if !strings.Contains(err.Error(), "if provided, maxExpirationSeconds must be <= 7862400") {
						framework.Failf("failed to create pod: %v", err)
					}
				}
			})

			ginkgo.It("should fail pod startup when MaxExpirationSeconds is less than minimum (1h)", func(ctx context.Context) {
				namespace := f.Namespace.Name
				ginkgo.By("Using namespace: " + namespace)

				// Less than 1 hour
				tooShortSeconds := int32(3599)
				testPod := createInspectorPod(namespace, "too-short-duration-pod", nil, &tooShortSeconds, spiffeSignerName)
				ginkgo.By("Creating pod requesting <1h MaxExpirationSeconds...")
				_, err := f.ClientSet.CoreV1().Pods(namespace).Create(ctx, testPod, metav1.CreateOptions{})
				if err != nil {
					if !strings.Contains(err.Error(), "if provided, maxExpirationSeconds must be >= 3600") {
						framework.Fail("Error message does not match the expected")
					}
				}
			})
		})
	})

func generateContainerSecurityContext() *v1.SecurityContext {
	desiredPSALevel := admissionapi.LevelRestricted
	desiredVersion := admissionapi.GetAPIVersion()
	minimalPod, err := admissiontest.GetMinimalValidPod(desiredPSALevel, desiredVersion)
	if err != nil {
		framework.Failf("failed to get minimal valid pod: %v", err)
	}
	return minimalPod.Spec.Containers[0].SecurityContext
}

// createServerObjects creates the Deployment and Service objects for the mTLS server.
func createServerObjects(namespace string, spiffeSignerName string, securityContext *v1.SecurityContext) (*appsv1.Deployment, *v1.Service) {
	replicas := int32(1)
	serverLabels := map[string]string{"app": "server"}

	signerNameVar := spiffeSignerName
	serverDeployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "server",
			Namespace: namespace,
			Labels:    serverLabels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: serverLabels},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: serverLabels},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "server",
							Image: imageutils.GetE2EImage(imageutils.Agnhost), // Use agnhost image >= 2.59
							Args: []string{
								"mtlsserver",
								"--listen=0.0.0.0:443",
								"--server-creds=/run/tls-config/spiffe-cred-bundle.pem",
								"--spiffe-trust-bundle=/run/tls-config/spiffe-trust-bundle.pem",
							},
							ImagePullPolicy: v1.PullAlways,
							SecurityContext: securityContext,
							VolumeMounts:    []v1.VolumeMount{{Name: "tls-config", MountPath: "/run/tls-config"}},
						},
					},
					Volumes: []v1.Volume{{
						Name: "tls-config",
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										PodCertificate: &v1.PodCertificateProjection{
											SignerName:           spiffeSignerName,
											CredentialBundlePath: "spiffe-cred-bundle.pem",
											KeyType:              "ECDSAP256",
										},
									},
									{
										ClusterTrustBundle: &v1.ClusterTrustBundleProjection{
											SignerName:    &signerNameVar,
											LabelSelector: &metav1.LabelSelector{},
											Path:          "spiffe-trust-bundle.pem",
										},
									},
								},
							},
						},
					}},
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
	}

	serverService := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "server", Namespace: namespace},
		Spec: v1.ServiceSpec{
			Type:     v1.ServiceTypeClusterIP,
			Ports:    []v1.ServicePort{{Name: "https", Port: 443}},
			Selector: serverLabels,
		},
	}

	return serverDeployment, serverService
}

// createClientObjects creates the Deployment object for the mTLS client.
func createClientObjects(namespace string, spiffeSignerName string, securityContext *v1.SecurityContext) *appsv1.Deployment {
	replicas := int32(1)
	clientLabels := map[string]string{"app": "client"}
	fetchURL := "https://server." + namespace + ".svc/spiffe-echo"
	signerNameVar := spiffeSignerName

	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "client",
			Namespace: namespace,
			Labels:    clientLabels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: clientLabels},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: clientLabels},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					Containers: []v1.Container{{
						Name:  "client",
						Image: imageutils.GetE2EImage(imageutils.Agnhost), // Use agnhost image >= 2.59
						Args: []string{
							"mtlsclient",
							"--fetch-url=" + fetchURL,
							"--server-trust-bundle=/run/tls-config/spiffe-trust-bundle.pem",
							"--client-cred-bundle=/run/tls-config/spiffe-cred-bundle.pem",
						},
						ImagePullPolicy: v1.PullAlways,
						SecurityContext: securityContext,
						VolumeMounts:    []v1.VolumeMount{{Name: "tls-config", MountPath: "/run/tls-config"}},
					}},
					Volumes: []v1.Volume{{
						Name: "tls-config",
						VolumeSource: v1.VolumeSource{
							Projected: &v1.ProjectedVolumeSource{
								Sources: []v1.VolumeProjection{
									{
										ClusterTrustBundle: &v1.ClusterTrustBundleProjection{
											SignerName:    &signerNameVar,
											LabelSelector: &metav1.LabelSelector{},
											Path:          "spiffe-trust-bundle.pem",
										},
									},
									{
										PodCertificate: &v1.PodCertificateProjection{
											SignerName:           spiffeSignerName,
											CredentialBundlePath: "spiffe-cred-bundle.pem",
											KeyType:              "ECDSAP256",
										},
									},
								},
							},
						},
					}},
				},
			},
		},
	}
}

// createInspectorPod creates a pod designed to print its certificate and wait, for inspection purposes.
func createInspectorPod(namespace, podName string, userAnnotations map[string]string, maxExpirationSeconds *int32, spiffeSignerName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "inspector",
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					// Use standard shell commands available in the agnhost base image
					Command: []string{"/bin/sh", "-c", "cat /run/tls-config/spiffe-cred-bundle.pem && sleep infinity"},
					VolumeMounts: []v1.VolumeMount{
						{Name: "tls-config", MountPath: "/run/tls-config"},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "tls-config",
					VolumeSource: v1.VolumeSource{
						Projected: &v1.ProjectedVolumeSource{
							Sources: []v1.VolumeProjection{
								{
									PodCertificate: &v1.PodCertificateProjection{
										SignerName:           spiffeSignerName,
										CredentialBundlePath: "spiffe-cred-bundle.pem",
										KeyType:              "ECDSAP256",
										UserAnnotations:      userAnnotations,
										MaxExpirationSeconds: maxExpirationSeconds,
									},
								},
							},
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}
