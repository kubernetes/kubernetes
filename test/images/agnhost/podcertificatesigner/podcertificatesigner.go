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

// Package podcertificatesigner is an agnhost subcommand implementing a toy
// PodCertificateRequest signer.  It is meant to run continuously in an
// in-cluster pod.
package podcertificatesigner

import (
	"bytes"
	"context"
	"crypto"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"flag"
	"fmt"
	"net/url"
	"os"
	"os/signal"
	"path"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	certsv1alpha1 "k8s.io/api/certificates/v1alpha1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	certinformersv1alpha1 "k8s.io/client-go/informers/certificates/v1alpha1"
	"k8s.io/client-go/kubernetes"
	certlistersv1alpha1 "k8s.io/client-go/listers/certificates/v1alpha1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/logs"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

var CmdPodCertificateSigner = &cobra.Command{
	Use:   "podcertificatesigner",
	Short: "Sign PodCertificateRequests addressed to a given signer",
	Args:  cobra.MaximumNArgs(0),
	RunE:  run,
}

var kubeconfigPath string
var signerName string

func init() {
	CmdPodCertificateSigner.Flags().StringVar(&kubeconfigPath, "kubeconfig", "", "Path to kubeconfig file to use for connection.  If omitted, in-cluster config will be used.")
	CmdPodCertificateSigner.Flags().StringVar(&signerName, "signer-name", "", "The signer name to sign certificates for")
}

func run(cmd *cobra.Command, args []string) error {
	flag.Set("logtostderr", "true")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	logs.InitLogs()
	defer logs.FlushLogs()

	cfg, err := clientcmd.BuildConfigFromFlags("", kubeconfigPath)
	if err != nil {
		return fmt.Errorf("while building client config: %w", err)
	}

	kc, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return fmt.Errorf("while creating kubernetes client: %w", err)
	}

	pcrInformer := certinformersv1alpha1.NewFilteredPodCertificateRequestInformer(kc, metav1.NamespaceAll, 24*time.Hour, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
		func(opts *metav1.ListOptions) {
			opts.FieldSelector = fields.OneTermEqualSelector("spec.signerName", signerName).String()
		},
	)

	caKeys, caCerts, err := generateCAHierarchy(1)
	if err != nil {
		return fmt.Errorf("while generating CA hierarchy: %w", err)
	}

	c := &controller{
		kc:          kc,
		pcrInformer: pcrInformer,
		pcrQueue:    workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[string]()),

		caKeys:  caKeys,
		caCerts: caCerts,
	}

	c.pcrInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(new any) {
			key, err := cache.MetaNamespaceKeyFunc(new)
			if err != nil {
				return
			}
			c.pcrQueue.Add(key)
		},
		UpdateFunc: func(old, new any) {
			key, err := cache.MetaNamespaceKeyFunc(new)
			if err != nil {
				return
			}
			c.pcrQueue.Add(key)
		},
		DeleteFunc: func(old any) {
			key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(old)
			if err != nil {
				return
			}
			c.pcrQueue.Add(key)
		},
	})

	go c.Run(ctx)

	// Wait for a shutdown signal.
	signalCh := make(chan os.Signal, 1)
	signal.Notify(signalCh, syscall.SIGINT, syscall.SIGTERM)
	<-signalCh

	// Canceling the context will begin exiting all of our controllers.
	cancel()

	return nil
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

type controller struct {
	kc          kubernetes.Interface
	pcrInformer cache.SharedIndexInformer
	pcrQueue    workqueue.TypedRateLimitingInterface[string]

	caKeys  []crypto.PrivateKey
	caCerts [][]byte
}

func (c *controller) Run(ctx context.Context) {
	defer c.pcrQueue.ShutDownWithDrain()
	go c.pcrInformer.Run(ctx.Done())
	if !cache.WaitForCacheSync(ctx.Done(), c.pcrInformer.HasSynced) {
		return
	}

	go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	<-ctx.Done()
}

func (c *controller) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

func (c *controller) processNextWorkItem(ctx context.Context) bool {
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

func (c *controller) handlePCR(ctx context.Context, pcr *certsv1alpha1.PodCertificateRequest) error {
	if pcr.Spec.SignerName != signerName {
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
