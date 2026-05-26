/*
Copyright 2026 The Kubernetes Authors.

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
package podcertssigner

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/asn1"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"strings"
	"sync"
	"time"

	certv1beta1 "k8s.io/api/certificates/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	certbeta1informers "k8s.io/client-go/informers/certificates/v1beta1"
	corev1informers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes"
	certv1beta1clients "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	certv1beta1listers "k8s.io/client-go/listers/certificates/v1beta1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

const (
	AddPodIPKey     = "kubernetes.io/add-pod-ip"
	AddPodFQDNKey   = "kubernetes.io/add-pod-fqdn"
	ServiceNamesKey = "kubernetes.io/service-names"
)

var userIDOID = asn1.ObjectIdentifier{0, 9, 2342, 19200300, 100, 1, 1}

type X509TemplateMapper interface {
	MapPodCertRequestToX509Template(*certv1beta1.PodCertificateRequest, *corev1.Pod) (*x509.Certificate, error)
}

type KubeServiceCATemplateMapper struct {
	ServiceLister corev1listers.ServiceLister
	ClusterDomain string
}

// TODO: deserializedDynamicCerts should likely be moved to dynamiccerts package
// as some sorts of DynamicCertKeyPairContent wrapper that won't allow direct
// access to the bytes so as to not confuse the caller in case these were different
// from the deserialized form due to deserialization errors
// NewDynamicSigningCertKeyPairContent() ?
// Optimize not to re-deserialize every time a sync is called but only on file
// content change.
type deserializedDynamicCerts struct {
	bytes *dynamiccertificates.DynamicCertKeyPairContent

	cert *x509.Certificate
	key  crypto.Signer

	queue workqueue.TypedRateLimitingInterface[struct{}]
}

func newDeserializedDynamicCerts(byteCertKeyPair *dynamiccertificates.DynamicCertKeyPairContent) (*deserializedDynamicCerts, error) {
	deserializedDynamicCerts := &deserializedDynamicCerts{
		bytes: byteCertKeyPair,
		queue: workqueue.NewTypedRateLimitingQueue(
			workqueue.DefaultTypedControllerRateLimiter[struct{}](),
		),
	}

	if err := deserializedDynamicCerts.loadCertKeyPair(); err != nil {
		return nil, err
	}
	deserializedDynamicCerts.bytes.AddListener(deserializedDynamicCerts)

	return deserializedDynamicCerts, nil
}

func (dc *deserializedDynamicCerts) Enqueue() {
	dc.queue.Add(struct{}{})
}

func (dc *deserializedDynamicCerts) Run(ctx context.Context) {
	defer utilruntime.HandleCrashWithContext(ctx)
	defer dc.queue.ShutDown()

	go dc.bytes.Run(ctx, 1)

	klog.InfoS("Starting deserialized dynamic certificates controller", "name", dc.bytes.Name())
	defer klog.InfoS("Shutting deserialized dynamic certificates", "name", dc.bytes.Name())

	// doesn't matter what workers say, only start one.
	go wait.Until(dc.runWorker, time.Second, ctx.Done())

	<-ctx.Done()
}

func (dc *deserializedDynamicCerts) runWorker() {
	for dc.processNextWorkItem() {
	}
}

func (dc *deserializedDynamicCerts) processNextWorkItem() bool {
	dsKey, quit := dc.queue.Get()
	if quit {
		return false
	}
	defer dc.queue.Done(dsKey)

	err := dc.loadCertKeyPair()
	if err == nil {
		dc.queue.Forget(dsKey)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", dsKey, err))
	dc.queue.AddRateLimited(dsKey)

	return true
}

func (dc *deserializedDynamicCerts) loadCertKeyPair() error {
	certBytes, keyBytes := dc.bytes.CurrentCertKeyContent()
	cert, err := certutil.ParseCertsPEM(certBytes)
	if err != nil {
		klog.ErrorS(err, "failed to parse certs", "controllerName", dc.bytes.Name())
		return nil // we don't want to retry until we get new content
	}

	key, err := keyutil.ParsePrivateKeyPEM(keyBytes)
	if err != nil {
		klog.ErrorS(err, "failed to parse keys", "controllerName", dc.bytes.Name())
		return nil // we don't want to retry until we get new content
	}
	privateKey, ok := key.(crypto.Signer)
	if !ok {
		klog.ErrorS(fmt.Errorf("failed to assert type of a crypto.PrivateKey"), "private key has an unexpected type", "privateKeyType", fmt.Sprintf("%T", key))
		return nil
	}

	dc.cert = cert[0] // TODO: which cert to choose? We may need more in the bundle for trust publishing in case we're rotating -> we can't require the bundle to contain a single cert
	dc.key = privateKey
	return nil
}

func (dc *deserializedDynamicCerts) CurrentCertKeyContent() (*x509.Certificate, crypto.Signer) {
	return dc.cert, dc.key
}

type PodCertsSignerController struct {
	signerName        string
	signerCertKeyPair *deserializedDynamicCerts

	podsLister              corev1listers.PodLister
	podCertRequestsInformer cache.SharedIndexInformer
	podCertRequestsLister   certv1beta1listers.PodCertificateRequestLister

	podCertRequestsClient certv1beta1clients.PodCertificateRequestsGetter

	podCertRequestsListerSynced func() bool
	podListerSynced             func() bool

	certTemplateMapper X509TemplateMapper

	queue workqueue.TypedRateLimitingInterface[types.NamespacedName]
}

func NewPodCertificatesSigner(
	ctx context.Context,
	signerName string,
	caCertPath string,
	caKeyPath string,
	certTemplateMapper X509TemplateMapper,
	kubeClient kubernetes.Interface,
	podInformers corev1informers.PodInformer,
) (*PodCertsSignerController, error) {
	certs, err := dynamiccertificates.NewDynamicServingContentFromFiles("service CA", caCertPath, caKeyPath) // FIXME: dynamiccerts should use properly named functions - these are not serving certs and the function allows this use
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic cert/key pair: %w", err)
	}
	if err := certs.RunOnce(ctx); err != nil {
		return nil, fmt.Errorf("failed to init certificate reloader: %w", err)
	}
	deserializedCerts, err := newDeserializedDynamicCerts(certs) // FIXME: the loadCertKeyPair() doesn't actually fail so the error here will be nil always, but cert/key may be nil!!! -> we may want to discard the queue key at all times on sync
	if err != nil {
		return nil, fmt.Errorf("failed to init deserializing certificate reloader: %w", err)
	}

	podCertRequestsInformer := certbeta1informers.NewFilteredPodCertificateRequestInformer(kubeClient, metav1.NamespaceAll, 0, cache.Indexers{},
		func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("spec.signerName", signerName).String()
		})

	c := &PodCertsSignerController{
		signerName:         signerName,
		signerCertKeyPair:  deserializedCerts,
		certTemplateMapper: certTemplateMapper,

		podsLister:              podInformers.Lister(),
		podCertRequestsInformer: podCertRequestsInformer,
		podCertRequestsLister:   certv1beta1listers.NewPodCertificateRequestLister(podCertRequestsInformer.GetIndexer()),

		podCertRequestsClient: kubeClient.CertificatesV1beta1(),

		podCertRequestsListerSynced: podCertRequestsInformer.HasSynced,
		podListerSynced:             podInformers.Informer().HasSynced,

		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[types.NamespacedName](),
			workqueue.TypedRateLimitingQueueConfig[types.NamespacedName]{
				Name: "pod_certs_request_signer", // TODO: should this be unique among all controllers of the same type?
			},
		),
	}

	podCertRequestsInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    func(obj any) { c.enqueue(obj) },
		UpdateFunc: func(oldObj, newObj any) { c.enqueue(newObj) }, // TODO: does this even make sense? When would the obj update, except for status? Does the status update matter?
	})

	return c, nil
}

func (s *PodCertsSignerController) enqueue(obj any) {
	pcr, ok := obj.(*certv1beta1.PodCertificateRequest)
	if !ok {
		panic(fmt.Errorf("unexpected object type: %T", obj))
	}

	if isPCRInFinalState(&pcr.Status) {
		return
	}

	objRef, err := cache.ObjectToName(pcr)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	s.queue.Add(objRef.AsNamespacedName())
}

func (s *PodCertsSignerController) Name() string {
	return s.signerName + "-signer" // FIXME maybe?
}

func (s *PodCertsSignerController) Run(ctx context.Context) {
	defer utilruntime.HandleCrashWithContext(ctx)

	logger := klog.FromContext(ctx)
	logger.Info("Starting PodCertsSignerController cert publisher controller", "signerName", s.signerName)

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down ClusterTrustBundle cert publisher controller", "signerName", s.signerName)
		s.queue.ShutDown()
		wg.Wait()
	}()

	wg.Go(func() { s.signerCertKeyPair.Run(ctx) })
	wg.Go(func() {
		s.podCertRequestsInformer.RunWithContext(ctx)
	})

	if !cache.WaitForNamedCacheSyncWithContext(ctx, s.podCertRequestsListerSynced, s.podListerSynced) {
		return
	}

	wg.Go(func() {
		wait.UntilWithContext(ctx, s.runWorker(), time.Second)
	})
	// FIXME: missing wg.Wait() ?
	<-ctx.Done()
}

func (s *PodCertsSignerController) runWorker() func(context.Context) {
	return func(ctx context.Context) {
		for s.processNextWorkItem(ctx) {
		}
	}
}

func (s *PodCertsSignerController) processNextWorkItem(ctx context.Context) bool {
	key, quit := s.queue.Get()
	if quit {
		return false
	}
	defer s.queue.Done(key)

	if err := s.syncPodCertificateRequest(ctx, key); err != nil {
		utilruntime.HandleError(fmt.Errorf("syncing %q failed: %w", key, err))
		s.queue.AddRateLimited(key)
		return true
	}

	s.queue.Forget(key)
	return true
}

func (s *PodCertsSignerController) syncPodCertificateRequest(ctx context.Context, key types.NamespacedName) error {
	// FIXME: should update PCR conditions in all the cases below

	certReq, err := s.podCertRequestsLister.PodCertificateRequests(key.Namespace).Get(key.Name)
	if err != nil {
		return fmt.Errorf("failed to fetch PodCertificateRequest %v: %w", key, err)
	}

	if isPCRInFinalState(&certReq.Status) {
		return nil
	}

	if certReq.DeletionTimestamp != nil {
		klog.V(5).InfoS("PodCertificateRequest is scheduled for deletion", "namespacedName", key)
		return nil
	}

	if certReq.Spec.SignerName != s.signerName {
		return fmt.Errorf("the PodCertificateRequest's signer name doesn't match the expected name - got: %q", certReq.Spec.SignerName)
	}

	requestingPod, err := s.podsLister.Pods(certReq.Namespace).Get(certReq.Spec.PodName)
	if err != nil {
		return fmt.Errorf("failed to find a pod requesting a cert via PodCertificateRequest '%v'", key)
	}

	if requestingPod.UID != certReq.Spec.PodUID {
		return fmt.Errorf("pod UIDs don't match for PodCertificateRequest '%v'", key)
	}
	// TODO: do we need to verify SA and Node parameters of the PCR, too?

	templateToSign, err := s.certTemplateMapper.MapPodCertRequestToX509Template(certReq, requestingPod)
	if err != nil {
		return err
	}

	caCert, caPrivKey := s.signerCertKeyPair.CurrentCertKeyContent()
	cert, err := x509.CreateCertificate(rand.Reader, templateToSign, caCert, templateToSign.PublicKey, caPrivKey)
	if err != nil {
		return fmt.Errorf("failed to sign cert: %w", err)
	}
	caPEMBytes, err := certutil.EncodeCertificates(caCert)
	if err != nil {
		return err
	}
	certChainPEMBytes := append(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert}), '\n')
	certChainPEMBytes = append(certChainPEMBytes, caPEMBytes...)

	certReqCopy := certReq.DeepCopy()
	certReqCopy.Status.CertificateChain = string(certChainPEMBytes)
	certReqCopy.Status.NotBefore = &metav1.Time{Time: templateToSign.NotBefore}
	certReqCopy.Status.NotAfter = &metav1.Time{Time: templateToSign.NotAfter}
	certReqCopy.Status.BeginRefreshAt = &metav1.Time{Time: templateToSign.NotAfter.Add(-15 * time.Minute)} // FIXME: should be at about 50, 60% of a cert lifetime?
	certReqCopy.Status.Conditions = []metav1.Condition{                                                    // FIXME: improve conditions
		{
			Type:               certv1beta1.PodCertificateRequestConditionTypeIssued,
			Status:             metav1.ConditionTrue,
			LastTransitionTime: metav1.NewTime(time.Now()),
			Reason:             "TwasAFineReq",
		},
	}

	if _, err := s.podCertRequestsClient.PodCertificateRequests(certReqCopy.Namespace).UpdateStatus(ctx, certReqCopy, metav1.UpdateOptions{}); err != nil {
		return err
	}

	return nil
}

func (m *KubeServiceCATemplateMapper) MapPodCertRequestToX509Template(pcr *certv1beta1.PodCertificateRequest, pod *corev1.Pod) (*x509.Certificate, error) {
	csr, err := x509.ParseCertificateRequest(pcr.Spec.StubPKCS10Request)
	if err != nil {
		return nil, fmt.Errorf("failed to parse CSR: %w", err)
	}

	// FIXME: improve the serial gen?
	maxSerial := new(big.Int).Lsh(big.NewInt(1), 159)
	var serial *big.Int
	for serial, err = rand.Int(rand.Reader, maxSerial); err != nil; serial, err = rand.Int(rand.Reader, maxSerial) {
	}

	issuanceTime := time.Now().Add(-2 * time.Minute)
	certLifetimeSeconds := max(3600, min(*pcr.Spec.MaxExpirationSeconds, 24*60*60))
	template := &x509.Certificate{
		SerialNumber: serial,
		Subject: pkix.Name{
			CommonName:   pcr.Spec.PodName,
			Organization: []string{pcr.Namespace},
			ExtraNames: []pkix.AttributeTypeAndValue{{
				Type:  userIDOID,
				Value: string(pcr.Spec.PodUID),
			}},
		},
		NotBefore:          issuanceTime,
		NotAfter:           issuanceTime.Add(time.Duration(certLifetimeSeconds) * time.Second),
		PublicKeyAlgorithm: csr.PublicKeyAlgorithm,
		PublicKey:          csr.PublicKey,
		ExtKeyUsage:        []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		KeyUsage:           x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature, // TODO: double check these
	}

	ipAddresses := []net.IP{} // FIXME: deduplicate IPs, keep where they come from for errors
	dnsNames := sets.New[string]()

	// TODO: additional source to go through: https://github.com/kubernetes/dns/blob/master/docs/specification.md
	if addPodIPVal := pcr.Spec.UnverifiedUserAnnotations[AddPodIPKey]; addPodIPVal == "true" {
		podIP := net.ParseIP(pod.Status.PodIP)
		if podIP == nil {
			return nil, fmt.Errorf("invalid pod IP: %s", podIP)
		}
		// TODO: should we add all IPs from pod.status.PodIPs?
		// --> probably, who are we to decide which IP to use
		ipAddresses = append(ipAddresses, podIP)
	}

	if addPodFQDNVal := pcr.Spec.UnverifiedUserAnnotations[AddPodFQDNKey]; addPodFQDNVal == "true" {
		// This branch is for Pods that get pointed to by a headless service. That typically
		// means the pod is a part of a StatefulSet.
		//
		// TODO: pod.Spec.setHostnameAsFQDN - https://kubernetes.io/docs/concepts/services-networking/dns-pod-service/#pod-sethostnameasfqdn-field
		// TODO: https://github.com/kubernetes/enhancements/issues/4762 - allow configuring any FQDN as pod's hostname

		subdomain := pod.Spec.Subdomain
		if len(subdomain) == 0 {
			return nil, fmt.Errorf("subdomain must be configured for pods requesting pod FQDN in certificates")
		}

		svc, err := m.ServiceLister.Services(pcr.Namespace).Get(subdomain)
		if err != nil {
			return nil, fmt.Errorf("couldn't get service %q for pod %q in namespace %q", subdomain, pod.Name, pcr.Namespace)
		}
		// FIXME: check the label selector of the service matches the pod!
		if !(svc.Spec.Type == corev1.ServiceTypeClusterIP && svc.Spec.ClusterIP == "None") {
			return nil, fmt.Errorf("service %q is not a headless service", svc.Name)
		}

		hostname := pod.Name
		if len(pod.Spec.Hostname) > 0 {
			hostname = pod.Spec.Hostname
		}
		fqdn := strings.Join(append([]string{hostname}, subdomain, pod.Namespace, "svc", m.ClusterDomain), ".")
		dnsNames.Insert(fqdn)
	}

	if serviceNames := pcr.Spec.UnverifiedUserAnnotations[ServiceNamesKey]; len(serviceNames) > 0 {
		serviceNamesSlice := strings.SplitSeq(serviceNames, ",")
		for svcName := range serviceNamesSlice {
			svc, err := m.ServiceLister.Services(pcr.Namespace).Get(svcName)
			if err != nil {
				return nil, fmt.Errorf("failed to fetch service %q: %w", svcName, err)
			}
			// TODO: make sure svc.Spec.Selector won't match everything if empty
			selector := labels.Set(svc.Spec.Selector).AsSelectorPreValidated()
			if !selector.Matches(labels.Set(pod.Labels)) {
				return nil, fmt.Errorf("pod %q does not match service %q selector", pod.Name, svc.Name)
			}

			// TODO: see if anything needs to be done for headless services
			if len(svc.Spec.ClusterIP) > 0 {
				svcIP := net.ParseIP(svc.Spec.ClusterIP)
				if svcIP == nil {
					return nil, fmt.Errorf("failed to parse %q service cluster IP", svc.Name) // TODO: add SVC NS in the error msg?
				}
				ipAddresses = append(ipAddresses, svcIP) // FIXME: add all IPs from svc.Spec.ClusterIPs if these are set
			}
			dnsNames.Insert(strings.Join([]string{svc.Name, svc.Namespace, "svc", m.ClusterDomain}, "."))
		}
	}

	if len(dnsNames) == 0 && len(ipAddresses) == 0 {
		// TODO: have package-specific errors that map to Failed/Denied/retriable?
		return nil, fmt.Errorf("the config from UnverifiedUserAnnotations would end up in a certificate without any SubjectAlternativeNames, thus being unusable for TLS serving")
	}

	template.DNSNames = sets.List(dnsNames)
	template.IPAddresses = ipAddresses

	return template, nil
}

func isPCRInFinalState(pcr *certv1beta1.PodCertificateRequestStatus) bool {
	for _, c := range pcr.Conditions {
		switch c.Type {
		case certv1beta1.PodCertificateRequestConditionTypeIssued,
			certv1beta1.PodCertificateRequestConditionTypeDenied,
			certv1beta1.PodCertificateRequestConditionTypeFailed:
			if c.Status == metav1.ConditionTrue {
				return true
			}
		}
	}
	return false
}
