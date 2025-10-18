/*
Copyright 2025 The Kubernetes Authors.

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

package kubelet

import (
	"context"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	certsv1beta1 "k8s.io/api/certificates/v1beta1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corelistersv1 "k8s.io/client-go/listers/core/v1"
	restclient "k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/kubelet/podcertificate"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/hermeticpodcertificatesigner"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
)

func TestPodCertificateManager(t *testing.T) {
	ctx, cancel := context.WithCancel(ktesting.Init(t))
	defer cancel()

	// Run an apiserver with PodCertificateRequest features enabled.
	s := kubeapiservertesting.StartTestServerOrDie(
		t,
		kubeapiservertesting.NewDefaultTestServerOptions(),
		[]string{
			"--authorization-mode=Node,RBAC",
			"--feature-gates=AuthorizeNodeWithSelectors=true,PodCertificateRequest=true",
			fmt.Sprintf("--runtime-config=%s=true", certsv1beta1.SchemeGroupVersion),
		},
		framework.SharedEtcd(),
	)
	defer s.TearDownFn()

	adminClient := kubernetes.NewForConfigOrDie(s.ClientConfig)

	var err error

	//
	// Configure and boot up a fake podcertificaterequest signing controller.
	//

	signerName := "foo.com/signer"

	signerSA, err := adminClient.CoreV1().ServiceAccounts("kube-system").Create(ctx, &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "kube-system",
			Name:      "foo-pcr-signing-controller",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating signer service account: %v", err)
	}

	signerClusterRole, err := adminClient.RbacV1().ClusterRoles().Create(ctx, &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo-com-pcr-signer",
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups:     []string{"certificates.k8s.io"},
				Resources:     []string{"signers"},
				Verbs:         []string{"sign"},
				ResourceNames: []string{"foo.com/*"},
			},
			{
				APIGroups: []string{"certificates.k8s.io"},
				Resources: []string{"podcertificaterequests"},
				Verbs:     []string{"get", "list", "watch"},
			},
			{
				APIGroups: []string{"certificates.k8s.io"},
				Resources: []string{"podcertificaterequests/status"},
				Verbs:     []string{"update"},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating signer ClusterRole: %v", err)
	}

	_, err = adminClient.RbacV1().ClusterRoleBindings().Create(ctx, &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "system:serviceaccount:kube-system:foo-pcr-signer",
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     signerClusterRole.Name,
		},
		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				Namespace: signerSA.ObjectMeta.Namespace,
				Name:      signerSA.ObjectMeta.Name,
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating signer ClusterRoleBinding: %v", err)
	}

	signerClient := mustServiceAccountClient(t, s.ClientConfig, signerSA.ObjectMeta.Namespace, signerSA.ObjectMeta.Name)
	caKeys, caCerts, err := hermeticpodcertificatesigner.GenerateCAHierarchy(1)
	if err != nil {
		t.Fatalf("Unexpected error generating CA hierarchy: %v", err)
	}
	pcrSigner := hermeticpodcertificatesigner.New(clock.RealClock{}, signerName, caKeys, caCerts, signerClient)
	go pcrSigner.Run(ctx)

	//
	// Configure and boot up enough Kubelet subsystems to run
	// podcertificate.IssuingManager.
	//
	node1, err := adminClient.CoreV1().Nodes().Create(ctx, &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating node1: %v", err)
	}

	node1Client := mustNodeClient(t, s.ClientConfig, node1.ObjectMeta.Name)
	node1PodInformerFactory := informers.NewSharedInformerFactoryWithOptions(node1Client, 0, informers.WithTweakListOptions(func(options *metav1.ListOptions) {
		options.FieldSelector = "spec.nodeName=" + node1.ObjectMeta.Name
	}))
	node1PCRInformerFactory := informers.NewSharedInformerFactoryWithOptions(node1Client, 0, informers.WithTweakListOptions(func(options *metav1.ListOptions) {
		options.FieldSelector = "spec.nodeName=" + node1.ObjectMeta.Name
	}))
	node1NodeInformerFactory := informers.NewSharedInformerFactoryWithOptions(node1Client, 0, informers.WithTweakListOptions(func(options *metav1.ListOptions) {
		options.FieldSelector = "metadata.name=" + node1.ObjectMeta.Name
	}))

	node1PodManager := &FakePodManager{
		podLister: node1PodInformerFactory.Core().V1().Pods().Lister(),
	}

	node1PodCertificateManager := podcertificate.NewIssuingManager(
		node1Client,
		node1PodManager,
		node1PCRInformerFactory.Certificates().V1beta1().PodCertificateRequests(),
		node1NodeInformerFactory.Core().V1().Nodes(),
		types.NodeName(node1.ObjectMeta.Name),
		clock.RealClock{},
	)

	node1PodInformerFactory.Start(ctx.Done())
	node1PCRInformerFactory.Start(ctx.Done())
	node1NodeInformerFactory.Start(ctx.Done())
	go node1PodCertificateManager.Run(ctx)

	//
	// Make a pod that uses a podcertificate volume.
	//

	workloadNS, err := adminClient.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "workload-ns",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating workload namespace: %v", err)
	}

	workloadSA, err := adminClient.CoreV1().ServiceAccounts(workloadNS.ObjectMeta.Name).Create(ctx, &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: workloadNS.ObjectMeta.Name,
			Name:      "workload",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating workload serviceaccount: %v", err)
	}

	workloadPod, err := adminClient.CoreV1().Pods(workloadNS.ObjectMeta.Name).Create(ctx, &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: workloadNS.ObjectMeta.Name,
			Name:      "workload",
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: workloadSA.ObjectMeta.Name,
			NodeName:           node1.ObjectMeta.Name,
			Containers: []corev1.Container{
				{
					Name:  "main",
					Image: "notarealimage",
					VolumeMounts: []corev1.VolumeMount{
						{
							Name:      "certificate",
							MountPath: "/run/foo-cert",
						},
					},
				},
			},
			Volumes: []corev1.Volume{
				{
					Name: "certificate",
					VolumeSource: corev1.VolumeSource{
						Projected: &corev1.ProjectedVolumeSource{
							Sources: []corev1.VolumeProjection{
								{
									PodCertificate: &corev1.PodCertificateProjection{
										SignerName:           signerName,
										KeyType:              "ED25519",
										CredentialBundlePath: "creds.pem",
										UserAnnotations:      map[string]string{hermeticpodcertificatesigner.SpiffePathKey: "workload"},
									},
								},
							},
						},
					},
				},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating workload pod: %v", err)
	}

	// Because our fake podManager is based on an informer, we need to poll
	// until workloadPod is reflected in the informer.
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		_, ok := node1PodManager.GetPodByUID(workloadPod.ObjectMeta.UID)
		return ok, nil
	})
	if err != nil {
		t.Fatalf("Error while waiting node1 podManager to know about workloadPod: %v", err)
	}

	node1PodCertificateManager.TrackPod(ctx, workloadPod)

	// Within a few seconds, we should see a PodCertificateRequest created for
	// this pod.
	var gotPCR *certsv1beta1.PodCertificateRequest
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		pcrs, err := adminClient.CertificatesV1beta1().PodCertificateRequests(workloadNS.ObjectMeta.Name).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, fmt.Errorf("while listing PodCertificateRequests: %w", err)
		}

		if len(pcrs.Items) == 0 {
			return false, nil
		}

		gotPCR = &pcrs.Items[0]
		return true, nil
	})
	if err != nil {
		t.Fatalf("Error while waiting for PCR to be created: %v", err)
	}

	// Check that the created PCR spec matches expectations.  Blank out fields on
	// gotPCR that we don't care about.  Blank out status, because the
	// controller might have already signed it.
	wantPCR := &certsv1beta1.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: workloadNS.ObjectMeta.Name,
		},
		Spec: certsv1beta1.PodCertificateRequestSpec{
			SignerName:                workloadPod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].PodCertificate.SignerName,
			PodName:                   workloadPod.ObjectMeta.Name,
			PodUID:                    workloadPod.ObjectMeta.UID,
			ServiceAccountName:        workloadSA.ObjectMeta.Name,
			ServiceAccountUID:         workloadSA.ObjectMeta.UID,
			NodeName:                  types.NodeName(node1.ObjectMeta.Name),
			NodeUID:                   node1.ObjectMeta.UID,
			MaxExpirationSeconds:      ptr.To[int32](86400),
			UnverifiedUserAnnotations: map[string]string{hermeticpodcertificatesigner.SpiffePathKey: "workload"},
		},
	}
	gotPCRClone := gotPCR.DeepCopy()
	gotPCRClone.ObjectMeta = metav1.ObjectMeta{}
	gotPCRClone.ObjectMeta.Namespace = gotPCR.ObjectMeta.Namespace
	gotPCRClone.Spec.PKIXPublicKey = nil
	gotPCRClone.Spec.ProofOfPossession = nil
	gotPCRClone.Status = certsv1beta1.PodCertificateRequestStatus{}
	if diff := cmp.Diff(gotPCRClone, wantPCR); diff != "" {
		t.Fatalf("PodCertificateManager created a bad PCR; diff (-got +want)\n%s", diff)
	}

	// Wait some more time for the PCR to be issued.
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		pcrs, err := adminClient.CertificatesV1beta1().PodCertificateRequests(workloadNS.ObjectMeta.Name).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, fmt.Errorf("while listing PodCertificateRequests: %w", err)
		}

		if len(pcrs.Items) == 0 {
			return false, nil
		}

		gotPCR = &pcrs.Items[0]

		for _, cond := range gotPCR.Status.Conditions {
			switch cond.Type {
			case certsv1beta1.PodCertificateRequestConditionTypeDenied,
				certsv1beta1.PodCertificateRequestConditionTypeFailed,
				certsv1beta1.PodCertificateRequestConditionTypeIssued:
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Error while waiting for PCR to be issued: %v", err)
	}

	isIssued := slices.ContainsFunc(gotPCR.Status.Conditions, func(cond metav1.Condition) bool {
		return cond.Type == certsv1beta1.PodCertificateRequestConditionTypeIssued
	})
	if !isIssued {
		t.Fatalf("The test signingController didn't issue the PCR:\n%+v", gotPCR)
	}

	// Check the spiffe path has been overridden with the UserAnnotations.
	issuedCertPem := []byte(gotPCR.Status.CertificateChain)
	block, _ := pem.Decode(issuedCertPem)
	cert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		t.Fatalf("Failed to parse the issued certificate: %v", err)
	}

	if cert.URIs[0].Path != "/workload" {
		t.Logf("Certificate path is %s", cert.URIs[0].Path)
		t.Fatalf("Failed to override the spiffe path with the user annotations")
	}
	// Now we know that the PCR was issued, so we can wait for the
	// podcertificate manager to return some valid credentials.
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		_, _, err := node1PodCertificateManager.GetPodCertificateCredentialBundle(ctx, workloadPod.ObjectMeta.Namespace, workloadPod.ObjectMeta.Name, string(workloadPod.ObjectMeta.UID), "certificate", 0)
		if err != nil {
			return false, err
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Error while waiting for podcertificate manager to return valid credentials: %v", err)
	}

	_, certChain, err := node1PodCertificateManager.GetPodCertificateCredentialBundle(ctx, workloadPod.ObjectMeta.Namespace, workloadPod.ObjectMeta.Name, string(workloadPod.ObjectMeta.UID), "certificate", 0)
	if err != nil {
		t.Fatalf("Unexpected error getting credentials from pod certificate manager: %v", err)
	}

	if diff := cmp.Diff(string(certChain), gotPCR.Status.CertificateChain); diff != "" {
		t.Fatalf("PodCertificate manager returned bad cert chain; diff (-got +want)\n%s", diff)
	}
}

type FakePodManager struct {
	podLister corelistersv1.PodLister
}

func (f *FakePodManager) GetPods() []*corev1.Pod {
	ret, _ := f.podLister.List(labels.Everything())
	return ret
}

func (f *FakePodManager) GetPodByUID(uid types.UID) (*corev1.Pod, bool) {
	list, err := f.podLister.List(labels.Everything())
	if err != nil {
		return nil, false
	}

	for _, pod := range list {
		if pod.ObjectMeta.UID == uid {
			return pod, true
		}
	}

	return nil, false
}

func mustServiceAccountClient(t *testing.T, cfg *restclient.Config, ns, sa string) *kubernetes.Clientset {
	newCfg := restclient.CopyConfig(cfg)
	newCfg.Impersonate.UserName = fmt.Sprintf("system:serviceaccount:%s:%s", ns, sa)
	newCfg.Impersonate.Groups = []string{"system:authenticated", "system:serviceaccounts"}
	kc, err := kubernetes.NewForConfig(newCfg)
	if err != nil {
		t.Fatalf("Unexpected error creating kubernetes client impersonating %q", newCfg.Impersonate.UserName)
	}
	return kc
}

func mustNodeClient(t *testing.T, cfg *restclient.Config, node string) *kubernetes.Clientset {
	newCfg := restclient.CopyConfig(cfg)
	newCfg.Impersonate.UserName = fmt.Sprintf("system:node:%s", node)
	newCfg.Impersonate.Groups = []string{"system:authenticated", "system:nodes"}
	kc, err := kubernetes.NewForConfig(newCfg)
	if err != nil {
		t.Fatalf("Unexpected error creating kubernetes client impersonating %q", newCfg.Impersonate.UserName)
	}
	return kc
}
