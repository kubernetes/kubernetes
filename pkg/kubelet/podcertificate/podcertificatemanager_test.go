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

package podcertificate

import (
	"context"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	certsv1alpha1 "k8s.io/api/certificates/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	corelistersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/test/utils/hermeticpodcertificatesigner"
	"k8s.io/kubernetes/test/utils/ktesting"
	testclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

func TestBasicFlow(t *testing.T) {
	ctx, cancel := context.WithCancel(ktesting.Init(t))
	defer cancel()

	kc := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactoryWithOptions(kc, 0)

	//
	// Configure and boot up a fake podcertificaterequest signing controller.
	//

	signerName := "foo.com/signer"

	caKeys, caCerts, err := hermeticpodcertificatesigner.GenerateCAHierarchy(1)
	if err != nil {
		t.Fatalf("Unexpected error generating CA hierarchy: %v", err)
	}
	pcrSigner := hermeticpodcertificatesigner.New(signerName, caKeys, caCerts, kc)
	go pcrSigner.Run(ctx)

	//
	// Configure and boot up enough Kubelet subsystems to run an IssuingManager.
	//

	node1, err := kc.CoreV1().Nodes().Create(ctx, &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating %s: %v", node1.ObjectMeta.Name, err)
	}

	node1PodManager := &FakePodManager{
		podLister: informerFactory.Core().V1().Pods().Lister(),
	}

	node1PodCertificateManager := NewIssuingManager(
		kc,
		node1PodManager,
		informerFactory.Certificates().V1alpha1().PodCertificateRequests(),
		informerFactory.Core().V1().Nodes(),
		types.NodeName(node1.ObjectMeta.Name),
		testclock.NewFakeClock(mustRFC3339(t, "2010-01-01T00:00:00Z")),
	)

	informerFactory.Start(ctx.Done())
	go node1PodCertificateManager.Run(ctx)

	//
	// Make a pod that uses a podcertificate volume.
	//

	workloadNS, err := kc.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: "workload-ns",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating workload namespace: %v", err)
	}

	workloadSA, err := kc.CoreV1().ServiceAccounts(workloadNS.ObjectMeta.Name).Create(ctx, &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: workloadNS.ObjectMeta.Name,
			Name:      "workload",
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error creating workload serviceaccount: %v", err)
	}

	workloadPod, err := kc.CoreV1().Pods(workloadNS.ObjectMeta.Name).Create(ctx, &corev1.Pod{
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
										MaxExpirationSeconds: ptr.To[int32](86400), // Defaulting doesn't work with a fake client.
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
		_, ok := node1PodManager.GetPodByName(workloadPod.ObjectMeta.Namespace, workloadPod.ObjectMeta.Name)
		return ok, nil
	})
	if err != nil {
		t.Fatalf("Error while waiting node1 podManager to know about workloadPod: %v", err)
	}

	node1PodCertificateManager.TrackPod(ctx, workloadPod)

	// Within a few seconds, we should see a PodCertificateRequest created for
	// this pod.
	var gotPCR *certsv1alpha1.PodCertificateRequest
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		pcrs, err := kc.CertificatesV1alpha1().PodCertificateRequests(workloadNS.ObjectMeta.Name).List(ctx, metav1.ListOptions{})
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
	wantPCR := &certsv1alpha1.PodCertificateRequest{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: workloadNS.ObjectMeta.Name,
		},
		Spec: certsv1alpha1.PodCertificateRequestSpec{
			SignerName:           workloadPod.Spec.Volumes[0].VolumeSource.Projected.Sources[0].PodCertificate.SignerName,
			PodName:              workloadPod.ObjectMeta.Name,
			PodUID:               workloadPod.ObjectMeta.UID,
			ServiceAccountName:   workloadSA.ObjectMeta.Name,
			ServiceAccountUID:    workloadSA.ObjectMeta.UID,
			NodeName:             types.NodeName(node1.ObjectMeta.Name),
			NodeUID:              node1.ObjectMeta.UID,
			MaxExpirationSeconds: ptr.To[int32](86400),
		},
	}
	gotPCRClone := gotPCR.DeepCopy()
	gotPCRClone.ObjectMeta = metav1.ObjectMeta{}
	gotPCRClone.ObjectMeta.Namespace = gotPCR.ObjectMeta.Namespace
	gotPCRClone.Spec.PKIXPublicKey = nil
	gotPCRClone.Spec.ProofOfPossession = nil
	gotPCRClone.Status = certsv1alpha1.PodCertificateRequestStatus{}
	if diff := cmp.Diff(gotPCRClone, wantPCR); diff != "" {
		t.Fatalf("PodCertificateManager created a bad PCR; diff (-got +want)\n%s", diff)
	}

	// Wait some more time for the PCR to be issued.
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		pcrs, err := kc.CertificatesV1alpha1().PodCertificateRequests(workloadNS.ObjectMeta.Name).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, fmt.Errorf("while listing PodCertificateRequests: %w", err)
		}

		if len(pcrs.Items) == 0 {
			return false, nil
		}

		gotPCR = &pcrs.Items[0]

		for _, cond := range gotPCR.Status.Conditions {
			switch cond.Type {
			case certsv1alpha1.PodCertificateRequestConditionTypeDenied,
				certsv1alpha1.PodCertificateRequestConditionTypeFailed,
				certsv1alpha1.PodCertificateRequestConditionTypeIssued:
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Error while waiting for PCR to be issued: %v", err)
	}

	isIssued := slices.ContainsFunc(gotPCR.Status.Conditions, func(cond metav1.Condition) bool {
		return cond.Type == certsv1alpha1.PodCertificateRequestConditionTypeIssued
	})
	if !isIssued {
		t.Fatalf("The test signingController didn't issue the PCR:\n%+v", gotPCR)
	}

	// Now we know that the PCR was issued, so we can wait for the
	// podcertificate manager to return some valid credentials.
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		_, _, err := node1PodCertificateManager.GetPodCertificateCredentialBundle(ctx, workloadPod.ObjectMeta.Namespace, workloadPod.ObjectMeta.Name, "certificate", 0)
		if err != nil {
			return false, err
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Error while waiting for podcertificate manager to return valid credentials: %v", err)
	}

	_, certChain, err := node1PodCertificateManager.GetPodCertificateCredentialBundle(ctx, workloadPod.ObjectMeta.Namespace, workloadPod.ObjectMeta.Name, "certificate", 0)
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

func (f *FakePodManager) GetPodByName(namespace, name string) (*corev1.Pod, bool) {
	ret, err := f.podLister.Pods(namespace).Get(name)
	if err != nil {
		return nil, false
	}
	return ret, true
}

func mustRFC3339(t *testing.T, stamp string) time.Time {
	got, err := time.Parse(time.RFC3339, stamp)
	if err != nil {
		t.Fatalf("Error while parsing timestamp: %v", err)
	}
	return got
}
