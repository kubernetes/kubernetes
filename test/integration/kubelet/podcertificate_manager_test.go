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
	"bytes"
	"context"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"net/url"
	"path"
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	certsv1alpha1 "k8s.io/api/certificates/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	certinformersv1alpha1 "k8s.io/client-go/informers/certificates/v1alpha1"
	"k8s.io/client-go/kubernetes"
	certlistersv1alpha1 "k8s.io/client-go/listers/certificates/v1alpha1"
	corelistersv1 "k8s.io/client-go/listers/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/kubelet/podcertificate"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
)

func TestPodCertificateManager(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Run an apiserver with PodCertificateRequest features enabled.
	s := kubeapiservertesting.StartTestServerOrDie(
		t,
		kubeapiservertesting.NewDefaultTestServerOptions(),
		[]string{
			"--authorization-mode=Node,RBAC",
			"--feature-gates=AuthorizeNodeWithSelectors=true,PodCertificateRequest=true",
			fmt.Sprintf("--runtime-config=%s=true", certsv1alpha1.SchemeGroupVersion),
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
	caKeys, caCerts, err := generateCAHierarchy(1)
	if err != nil {
		t.Fatalf("Unexpected error generating CA hierarchy: %v", err)
	}
	pcrSigner := newSigningController(signerName, caKeys, caCerts, signerClient)
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
		t.Fatalf("Unexpected error creating %s: %v", node1.ObjectMeta.Name, err)
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
		node1PCRInformerFactory.Certificates().V1alpha1().PodCertificateRequests(),
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

	err = node1PodCertificateManager.TrackPod(ctx, workloadPod)
	if err != nil {
		t.Fatalf("Unexpected error calling TrackPod: %v", err)
	}

	// Within a few seconds, we should see a PodCertificateRequest created for
	// this pod.
	var gotPCR *certsv1alpha1.PodCertificateRequest
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
		pcrs, err := adminClient.CertificatesV1alpha1().PodCertificateRequests(workloadNS.ObjectMeta.Name).List(ctx, metav1.ListOptions{})
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

	// Check that the created PCR spec matches expections.  Blank out fields on
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
		pcrs, err := adminClient.CertificatesV1alpha1().PodCertificateRequests(workloadNS.ObjectMeta.Name).List(ctx, metav1.ListOptions{})
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
		t.Fatalf("Error while waiting for PCR to be created: %v", err)
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
		_, _, err := node1PodCertificateManager.GetPodCertificateCredentialBundle(ctx, workloadPod.ObjectMeta.UID, "certificate", 0)
		if err != nil {
			return false, err
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("Error while waiting for podcertificate manager to return valid credentials: %v", err)
	}

	_, certChain, err := node1PodCertificateManager.GetPodCertificateCredentialBundle(ctx, workloadPod.ObjectMeta.UID, "certificate", 0)
	if err != nil {
		t.Fatalf("Unexpected error getting credentials from pod certificate manager: %v", err)
	}

	if diff := cmp.Diff(string(certChain), gotPCR.Status.CertificateChain); diff != "" {
		t.Fatalf("PodCertificate manager returned bad cert chain; diff (-got +want)\n%s", diff)
	}
}

type signingController struct {
	signerName string

	kc          kubernetes.Interface
	pcrInformer cache.SharedIndexInformer
	pcrQueue    workqueue.TypedRateLimitingInterface[string]

	caKeys  []crypto.PrivateKey
	caCerts [][]byte
}

func newSigningController(signerName string, caKeys []crypto.PrivateKey, caCerts [][]byte, kc kubernetes.Interface) *signingController {
	pcrInformer := certinformersv1alpha1.NewFilteredPodCertificateRequestInformer(kc, metav1.NamespaceAll, 24*time.Hour, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		func(opts *metav1.ListOptions) {
			opts.FieldSelector = fields.OneTermEqualSelector("spec.signerName", signerName).String()
		},
	)

	sc := &signingController{
		signerName:  signerName,
		kc:          kc,
		pcrInformer: pcrInformer,
		pcrQueue:    workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]()),
		caKeys:      caKeys,
		caCerts:     caCerts,
	}

	sc.pcrInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(new any) {
			key, err := cache.MetaNamespaceKeyFunc(new)
			if err != nil {
				return
			}
			sc.pcrQueue.Add(key)
		},
		UpdateFunc: func(old, new any) {
			key, err := cache.MetaNamespaceKeyFunc(new)
			if err != nil {
				return
			}
			sc.pcrQueue.Add(key)
		},
		DeleteFunc: func(old any) {
			key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(old)
			if err != nil {
				return
			}
			sc.pcrQueue.Add(key)
		},
	})

	return sc
}

func (c *signingController) Run(ctx context.Context) {
	defer c.pcrQueue.ShutDown()
	go c.pcrInformer.Run(ctx.Done())
	if !cache.WaitForCacheSync(ctx.Done(), c.pcrInformer.HasSynced) {
		return
	}

	go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	<-ctx.Done()
}

func (c *signingController) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *signingController) processNextWorkItem(ctx context.Context) bool {
	key, quit := c.pcrQueue.Get()
	if quit {
		return false
	}
	defer c.pcrQueue.Done(key)

	klog.InfoS("Processing PCR", "key", key)

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		klog.ErrorS(err, "Error while splitting key into namespace and name", "key", key)
		return true
	}

	pcr, err := certlistersv1alpha1.NewPodCertificateRequestLister(c.pcrInformer.GetIndexer()).PodCertificateRequests(namespace).Get(name)
	if k8serrors.IsNotFound(err) {
		c.pcrQueue.Forget(key)
		return true
	} else if err != nil {
		klog.ErrorS(err, "Error while retrieving PodCertificateRequest", "key", key)
		return true
	}

	err = c.handlePCR(ctx, pcr)
	if err != nil {
		klog.ErrorS(err, "Error while handling PodCertificateRequest", "key", key)
		c.pcrQueue.AddRateLimited(key)
		return true
	}

	c.pcrQueue.Forget(key)
	return true
}

func (c *signingController) handlePCR(ctx context.Context, pcr *certsv1alpha1.PodCertificateRequest) error {
	if pcr.Spec.SignerName != c.signerName {
		return nil
	}

	// PodCertificateRequests don't have an approval stage, and the node
	// restriction / isolation check is handled by kube-apiserver.

	// If our signer had a policy about which pods are allowed to request
	// certificates, it would be implemented here.

	// Proceed to signing.  Our toy signer will make a SPIFFE cert encoding the
	// namespace and name of the pod's service account.

	// Is the PCR already signed?
	if pcr.Status.CertificateChain != "" {
		return nil
	}

	subjectPublicKey, err := x509.ParsePKIXPublicKey(pcr.Spec.PKIXPublicKey)
	if err != nil {
		return fmt.Errorf("while parsing subject public key: %w", err)
	}

	// If our signer had an opinion on which key types were allowable, it would
	// check subjectPublicKey, and deny the PCR with a SuggestedKeyType
	// condition on it.

	lifetime := 24 * time.Hour
	if pcr.Spec.MaxExpirationSeconds != nil {
		requestedLifetime := time.Duration(*pcr.Spec.MaxExpirationSeconds) * time.Second
		if requestedLifetime < lifetime {
			lifetime = requestedLifetime
		}
	}

	spiffeURI := &url.URL{
		Scheme: "spiffe",
		Host:   "cluster.local",
		Path:   path.Join("ns", pcr.ObjectMeta.Namespace, "sa", pcr.Spec.ServiceAccountName),
	}

	notBefore := time.Now().Add(-2 * time.Minute)
	notAfter := notBefore.Add(lifetime)
	beginRefreshAt := notAfter.Add(-30 * time.Minute)
	template := &x509.Certificate{
		URIs:        []*url.URL{spiffeURI},
		NotBefore:   notBefore,
		NotAfter:    notAfter,
		KeyUsage:    x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
	}

	signingCert, err := x509.ParseCertificate(c.caCerts[len(c.caCerts)-1])
	if err != nil {
		return fmt.Errorf("while parsing signing certificate: %w", err)
	}

	subjectCertDER, err := x509.CreateCertificate(rand.Reader, template, signingCert, subjectPublicKey, c.caKeys[len(c.caKeys)-1])
	if err != nil {
		return fmt.Errorf("while signing subject cert: %w", err)
	}

	// Compose the certificate chain
	chainPEM := &bytes.Buffer{}
	err = pem.Encode(chainPEM, &pem.Block{
		Type:  "CERTIFICATE",
		Bytes: subjectCertDER,
	})
	if err != nil {
		return fmt.Errorf("while encoding leaf certificate to PEM: %w", err)
	}
	for i := 0; i < len(c.caCerts)-1; i++ {
		err = pem.Encode(chainPEM, &pem.Block{
			Type:  "CERTIFICATE",
			Bytes: c.caCerts[len(c.caCerts)-1-i],
		})
		if err != nil {
			return fmt.Errorf("while encoding intermediate certificate to PEM: %w", err)
		}
	}

	// Don't modify the copy in the informer cache.
	pcr = pcr.DeepCopy()
	pcr.Status.Conditions = []metav1.Condition{
		{
			Type:   certsv1alpha1.PodCertificateRequestConditionTypeIssued,
			Status: metav1.ConditionTrue,
		},
	}
	pcr.Status.CertificateChain = chainPEM.String()
	pcr.Status.NotBefore = ptr.To(metav1.NewTime(notBefore))
	pcr.Status.BeginRefreshAt = ptr.To(metav1.NewTime(beginRefreshAt))
	pcr.Status.NotAfter = ptr.To(metav1.NewTime(notAfter))

	_, err = c.kc.CertificatesV1alpha1().PodCertificateRequests(pcr.ObjectMeta.Namespace).UpdateStatus(ctx, pcr, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("while updating PodCertificateRequest: %w", err)
	}

	return nil
}

type FakePodManager struct {
	podLister corelistersv1.PodLister
}

func (f *FakePodManager) GetPods() []*corev1.Pod {
	ret, _ := f.podLister.List(labels.Everything())
	return ret
}

func generateCAHierarchy(numIntermediates int) ([]crypto.PrivateKey, [][]byte, error) {
	caKeys := []crypto.PrivateKey{}
	caCerts := [][]byte{}

	rootPubKey, rootPrivKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, nil, fmt.Errorf("while generating root key: %w", err)
	}

	notBefore := time.Now()
	notAfter := notBefore.Add(365 * 24 * time.Hour)

	rootTemplate := &x509.Certificate{
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		IsCA:                  true,
		BasicConstraintsValid: true,
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
	}

	rootDER, err := x509.CreateCertificate(rand.Reader, rootTemplate, rootTemplate, rootPubKey, rootPrivKey)
	if err != nil {
		return nil, nil, fmt.Errorf("while generating root certificate: %w", err)
	}

	caKeys = append(caKeys, rootPrivKey)
	caCerts = append(caCerts, rootDER)

	for i := 0; i < numIntermediates; i++ {
		pubKey, privKey, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			return nil, nil, fmt.Errorf("while generating intermediate key: %w", err)
		}

		template := &x509.Certificate{
			NotBefore:             notBefore,
			NotAfter:              notAfter,
			IsCA:                  true,
			BasicConstraintsValid: true,
			KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		}

		signingCert, err := x509.ParseCertificate(caCerts[len(caCerts)-1])
		if err != nil {
			return nil, nil, fmt.Errorf("while parsing previous cert: %w", err)
		}

		intermediateDER, err := x509.CreateCertificate(rand.Reader, template, signingCert, pubKey, caKeys[len(caCerts)-1])
		if err != nil {
			return nil, nil, fmt.Errorf("while signing intermediate certificate: %w", err)
		}

		caKeys = append(caKeys, privKey)
		caCerts = append(caCerts, intermediateDER)
	}

	return caKeys, caCerts, nil
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
