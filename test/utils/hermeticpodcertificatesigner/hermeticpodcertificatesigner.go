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

package hermeticpodcertificatesigner

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
	"strings"
	"time"

	certsv1beta1 "k8s.io/api/certificates/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	certinformersv1beta1 "k8s.io/client-go/informers/certificates/v1beta1"
	"k8s.io/client-go/kubernetes"
	certlistersv1beta1 "k8s.io/client-go/listers/certificates/v1beta1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
)

const SpiffePathKey = "spiffe/path-overriding"

// Controller is an in-memory signing controller for PodCertificateRequests.
type Controller struct {
	clock      clock.PassiveClock
	signerName string

	kc          kubernetes.Interface
	pcrInformer cache.SharedIndexInformer
	pcrQueue    workqueue.TypedRateLimitingInterface[string]
	pcrLister   certlistersv1beta1.PodCertificateRequestLister
	caKeys      []crypto.PrivateKey
	caCerts     [][]byte
}

// New creates a new Controller.
func New(clock clock.PassiveClock, signerName string, caKeys []crypto.PrivateKey, caCerts [][]byte, kc kubernetes.Interface) *Controller {
	pcrInformer := certinformersv1beta1.NewFilteredPodCertificateRequestInformer(kc, metav1.NamespaceAll, 24*time.Hour, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		func(opts *metav1.ListOptions) {
			opts.FieldSelector = fields.OneTermEqualSelector("spec.signerName", signerName).String()
		},
	)

	sc := &Controller{
		clock:       clock,
		signerName:  signerName,
		kc:          kc,
		pcrInformer: pcrInformer,
		pcrQueue:    workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]()),
		pcrLister:   certlistersv1beta1.NewPodCertificateRequestLister(pcrInformer.GetIndexer()),
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

func (c *Controller) Run(ctx context.Context) {
	defer c.pcrQueue.ShutDown()
	prefix := strings.Replace(c.signerName, "/", ":", 1)
	ctbName := prefix + ":primary-bundle"
	defer func() {
		klog.Infof("Deleting ClusterTrustBundle %s", ctbName)
		err := c.kc.CertificatesV1beta1().ClusterTrustBundles().Delete(context.Background(), ctbName, metav1.DeleteOptions{})
		if err != nil && !k8serrors.IsNotFound(err) {
			klog.Errorf("Failed to delete ClusterTrustBundle %s: %v", ctbName, err)
		}
	}()

	go c.pcrInformer.Run(ctx.Done())
	if !cache.WaitForCacheSync(ctx.Done(), c.pcrInformer.HasSynced) {
		return
	}

	go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	go wait.JitterUntilWithContext(ctx, c.ensureTrustBundle, 1*time.Minute, 1.0, true)
	<-ctx.Done()
}

func (c *Controller) ensureTrustBundle(ctx context.Context) {
	// Create a ClusterTrustBundle with the signer's CA.
	prefix := strings.Replace(c.signerName, "/", ":", 1)
	ctbName := prefix + ":primary-bundle"
	caCertPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: c.caCerts[0]})
	wantCTB := &certsv1beta1.ClusterTrustBundle{
		ObjectMeta: metav1.ObjectMeta{Name: ctbName},
		Spec: certsv1beta1.ClusterTrustBundleSpec{
			SignerName:  c.signerName,
			TrustBundle: string(caCertPEM),
		},
	}

	klog.Infof("Getting ClusterTrustBundle %s", ctbName)
	ctb, err := c.kc.CertificatesV1beta1().ClusterTrustBundles().Get(ctx, wantCTB.ObjectMeta.Name, metav1.GetOptions{})
	if k8serrors.IsNotFound(err) {
		_, err := c.kc.CertificatesV1beta1().ClusterTrustBundles().Create(ctx, wantCTB, metav1.CreateOptions{})
		if err != nil {
			klog.Errorf("Failed to create ClusterTrustBundle %s: %v", ctbName, err)
			return
		}
		return
	} else if err != nil {
		klog.Errorf("Failed to get ClusterTrustBundle %s: %v", ctbName, err)
		return
	}

	if apiequality.Semantic.DeepEqual(wantCTB.Spec, ctb.Spec) {
		klog.Info("ClusterTrustBundle already in correct state")
		return
	}

	ctb = ctb.DeepCopy()
	ctb.Spec = wantCTB.Spec

	_, err = c.kc.CertificatesV1beta1().ClusterTrustBundles().Update(ctx, ctb, metav1.UpdateOptions{})
	if err != nil {
		klog.Errorf("Failed to update ClusterTrustBundle %s: %v", ctbName, err)
	}
}

func (c *Controller) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *Controller) processNextWorkItem(ctx context.Context) bool {
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

	pcr, err := c.pcrLister.PodCertificateRequests(namespace).Get(name)
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

func (c *Controller) handlePCR(ctx context.Context, pcr *certsv1beta1.PodCertificateRequest) error {
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
	requestedLifetime := time.Duration(*pcr.Spec.MaxExpirationSeconds) * time.Second
	if requestedLifetime < lifetime {
		lifetime = requestedLifetime
	}
	path := path.Join("ns", pcr.ObjectMeta.Namespace, "sa", pcr.Spec.ServiceAccountName)
	if pcr.Spec.UnverifiedUserAnnotations != nil {
		if value, exist := pcr.Spec.UnverifiedUserAnnotations[SpiffePathKey]; exist {
			path = value
		}
	}
	spiffeURI := &url.URL{
		Scheme: "spiffe",
		Host:   "cluster.local",
		Path:   path,
	}

	notBefore := c.clock.Now().Add(-2 * time.Minute)
	notAfter := notBefore.Add(lifetime)
	beginRefreshAt := notAfter.Add(-30 * time.Minute)
	// Construct DNS names
	dnsNames := []string{
		fmt.Sprintf("server.%s.svc", pcr.ObjectMeta.Namespace),
	}
	template := &x509.Certificate{
		URIs:        []*url.URL{spiffeURI},
		NotBefore:   notBefore,
		NotAfter:    notAfter,
		KeyUsage:    x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		DNSNames:    dnsNames,
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
			Type:               certsv1beta1.PodCertificateRequestConditionTypeIssued,
			Status:             metav1.ConditionTrue,
			Reason:             "Reason",
			Message:            "Issued",
			LastTransitionTime: metav1.NewTime(c.clock.Now()),
		},
	}
	pcr.Status.CertificateChain = chainPEM.String()
	pcr.Status.NotBefore = ptr.To(metav1.NewTime(notBefore))
	pcr.Status.BeginRefreshAt = ptr.To(metav1.NewTime(beginRefreshAt))
	pcr.Status.NotAfter = ptr.To(metav1.NewTime(notAfter))

	_, err = c.kc.CertificatesV1beta1().PodCertificateRequests(pcr.ObjectMeta.Namespace).UpdateStatus(ctx, pcr, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("while updating PodCertificateRequest: %w", err)
	}
	return nil
}

// GenerateCAHierarchy makes a CA hierarchy, possibly with intermediates.  The
// outputs can be used with Controller.
func GenerateCAHierarchy(numIntermediates int) ([]crypto.PrivateKey, [][]byte, error) {
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
