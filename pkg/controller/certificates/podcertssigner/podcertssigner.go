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
	"math"
	"math/big"
	"strings"
	"sync"
	"time"

	certv1 "k8s.io/api/certificates/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	certv1informers "k8s.io/client-go/informers/certificates/v1"
	corev1informers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes"
	certv1clients "k8s.io/client-go/kubernetes/typed/certificates/v1"
	corev1clients "k8s.io/client-go/kubernetes/typed/core/v1"
	certv1listers "k8s.io/client-go/listers/certificates/v1"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

const (
	AddPodFQDNKey   = "kubernetes.io/add-pod-fqdn"
	ServiceNamesKey = "kubernetes.io/service-names"
)

var userIDOID = asn1.ObjectIdentifier{0, 9, 2342, 19200300, 100, 1, 1}

// X509TemplateMapper is used to map PodCertificateRequest object to a *x509.Certificate
// that will used to sign the resulting certificate.
type X509TemplateMapper interface {
	// HookEvents serves for the mapper to be able to add items to the controller's
	// workqueue in case it depends on some external conditions that might only
	// be fullfiled after a PCR was created.
	// This helps us deal with Kubernetes' eventual consistency.
	HookEvents(workqueue.TypedRateLimitingInterface[types.NamespacedName])
	MapPodCertRequestToX509Template(context.Context, *certv1.PodCertificateRequest, *corev1.Pod) (*x509.Certificate, error)
}

type KubeServiceCATemplateMapper struct {
	ServiceLister   corev1listers.ServiceLister
	ServiceInformer cache.SharedIndexInformer
	ServiceClient   corev1clients.ServicesGetter // TODO: wire

	ClusterDomain string

	missingServicesLock sync.RWMutex
	missingServices     map[types.NamespacedName][]types.NamespacedName
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
		return fmt.Errorf("failed to parse certs: %w", err)
	}

	key, err := keyutil.ParsePrivateKeyPEM(keyBytes)
	if err != nil {
		return fmt.Errorf("failed to parse keys: %w", err)
	}

	privateKey, ok := key.(crypto.Signer)
	if !ok {
		return fmt.Errorf("failed to assert type of a crypto.PrivateKey, key is of type %T", key)
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
	podCertRequestsLister   certv1listers.PodCertificateRequestLister

	podCertRequestsClient certv1clients.PodCertificateRequestsGetter

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
	deserializedCerts, err := newDeserializedDynamicCerts(certs)
	if err != nil {
		return nil, fmt.Errorf("failed to init deserializing certificate reloader: %w", err)
	}

	podCertRequestsInformer := certv1informers.NewFilteredPodCertificateRequestInformer(kubeClient, metav1.NamespaceAll, 0, cache.Indexers{},
		func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("spec.signerName", signerName).String()
		})

	c := &PodCertsSignerController{
		signerName:         signerName,
		signerCertKeyPair:  deserializedCerts,
		certTemplateMapper: certTemplateMapper,

		podsLister:              podInformers.Lister(),
		podCertRequestsInformer: podCertRequestsInformer,
		podCertRequestsLister:   certv1listers.NewPodCertificateRequestLister(podCertRequestsInformer.GetIndexer()),

		podCertRequestsClient: kubeClient.CertificatesV1(),

		podCertRequestsListerSynced: podCertRequestsInformer.HasSynced,
		podListerSynced:             podInformers.Informer().HasSynced,

		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[types.NamespacedName](),
			workqueue.TypedRateLimitingQueueConfig[types.NamespacedName]{
				Name: "pod_certs_request_signer", // TODO: should this be unique among all controllers of the same type?
			},
		),
	}
	certTemplateMapper.HookEvents(c.queue)

	podCertRequestsInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		// PCR objects are immutable so we only really ever care about them being added
		AddFunc: func(obj any) { c.enqueue(obj) },
	})

	return c, nil
}

func (s *PodCertsSignerController) enqueue(obj any) {
	pcr, ok := obj.(*certv1.PodCertificateRequest)
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
	logger.Info("Starting PodCertsSignerController certificate signing controller", "signerName", s.signerName)

	var wg sync.WaitGroup
	defer func() {
		logger.Info("Shutting down PodCertsSignerController certificate signing controller", "signerName", s.signerName)
		s.queue.ShutDown()
		wg.Wait()
	}()

	wg.Go(func() { s.signerCertKeyPair.Run(ctx) })
	wg.Go(func() { s.podCertRequestsInformer.RunWithContext(ctx) })

	if !cache.WaitForNamedCacheSyncWithContext(ctx, s.podCertRequestsListerSynced, s.podListerSynced) {
		return
	}

	wg.Go(func() { wait.UntilWithContext(ctx, s.runWorker(), time.Second) })
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

// syncPodCertificate request processes a PodCertificate request and mints a certificate
// if the request is valid.
//
// On success, it updates the PodCertificate's "Issued" condition to true.
// On errors, or when the request is not considered valid, it errors out and goes
// into error backoff. It must not set "Failed" or "Denied" conditions as that would
// make the pod be forever stuck in pending state as kubelet won't try requesting
// new pod certificates at that point.
func (s *PodCertsSignerController) syncPodCertificateRequest(ctx context.Context, key types.NamespacedName) error {
	certReq, err := s.podCertRequestsLister.PodCertificateRequests(key.Namespace).Get(key.Name)
	if errors.IsNotFound(err) {
		return nil
	} else if err != nil {
		return fmt.Errorf("failed to fetch PodCertificateRequest %v: %w", key, err)
	}

	if certReq.DeletionTimestamp != nil {
		klog.V(5).InfoS("PodCertificateRequest is scheduled for deletion", "namespacedName", key)
		return nil
	}

	if isPCRInFinalState(&certReq.Status) {
		return nil
	}

	// this would mean a bug in the informer setup
	if certReq.Spec.SignerName != s.signerName {
		return fmt.Errorf("the PodCertificateRequest's signer name doesn't match the expected name - got: %q", certReq.Spec.SignerName)
	}

	// make sure we've got the correct pod
	requestingPod, err := s.podsLister.Pods(certReq.Namespace).Get(certReq.Spec.PodName)
	if err != nil {
		return fmt.Errorf("failed to find a pod requesting a cert via PodCertificateRequest '%v'", key)
	}

	if requestingPod.UID != certReq.Spec.PodUID {
		return fmt.Errorf("pod UIDs don't match for PodCertificateRequest '%v'", key)
	}

	templateToSign, err := s.certTemplateMapper.MapPodCertRequestToX509Template(ctx, certReq, requestingPod)
	if err != nil || templateToSign == nil {
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
	certReqCopy.Status.Conditions = []metav1.Condition{
		{
			Type:               certv1.PodCertificateRequestConditionTypeIssued,
			Status:             metav1.ConditionTrue,
			Reason:             "ValidRequest",
			LastTransitionTime: metav1.NewTime(time.Now()),
			Message:            "All is well",
		},
	}

	if _, err := s.podCertRequestsClient.PodCertificateRequests(certReqCopy.Namespace).UpdateStatus(ctx, certReqCopy, metav1.UpdateOptions{}); err != nil {
		return err
	}

	return nil
}

func NewKubeServiceCATemplateMapper(
	serviceClient corev1clients.ServicesGetter,
	serviceInformer cache.SharedIndexInformer,
	serviceLister corev1listers.ServiceLister,
	clusterDomain string,
) *KubeServiceCATemplateMapper {
	return &KubeServiceCATemplateMapper{
		ServiceClient:   serviceClient,
		ServiceInformer: serviceInformer,
		ServiceLister:   serviceLister,
		ClusterDomain:   clusterDomain,

		missingServicesLock: sync.RWMutex{},
		missingServices:     map[types.NamespacedName][]types.NamespacedName{},
	}
}

func (m *KubeServiceCATemplateMapper) HookEvents(q workqueue.TypedRateLimitingInterface[types.NamespacedName]) {
	m.ServiceInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			svc := obj.(metav1.ObjectMetaAccessor)
			objMeta := svc.GetObjectMeta()

			m.missingServicesLock.Lock()
			defer m.missingServicesLock.Unlock()
			pcrs, ok := m.missingServices[types.NamespacedName{Namespace: objMeta.GetNamespace(), Name: objMeta.GetName()}]
			if !ok {
				return
			}
			for _, pcrKey := range pcrs {
				q.AddRateLimited(pcrKey)
				delete(m.missingServices, types.NamespacedName{Namespace: objMeta.GetNamespace(), Name: objMeta.GetName()})
			}
		},
		UpdateFunc: func(oldObj, newObj any) {
			oldSvc := oldObj.(*corev1.Service)
			newSvc := newObj.(*corev1.Service)

			// we really only care about type and selector changes as that make the difference
			// in whether pods matched or not
			if oldSvc.Spec.Type == newSvc.Spec.Type &&
				equality.Semantic.DeepEqual(oldSvc.Spec.Selector, newSvc.Spec.Selector) {
				return
			}

			m.missingServicesLock.Lock()
			defer m.missingServicesLock.Unlock()
			pcrs, ok := m.missingServices[types.NamespacedName{Namespace: newSvc.Namespace, Name: newSvc.Name}]
			if !ok {
				return
			}
			for _, pcrKey := range pcrs {
				q.AddRateLimited(pcrKey)
				delete(m.missingServices, types.NamespacedName{Namespace: newSvc.Namespace, Name: newSvc.Name})
			}
		},
	})
}

func (m *KubeServiceCATemplateMapper) MapPodCertRequestToX509Template(ctx context.Context, pcr *certv1.PodCertificateRequest, pod *corev1.Pod) (*x509.Certificate, error) {
	if !m.ServiceInformer.HasSynced() {
		if !cache.WaitForNamedCacheSyncWithContext(ctx, m.ServiceInformer.HasSynced) {
			return nil, fmt.Errorf("failed to wait for services cache sync")
		}
	}

	logger := klog.LoggerWithName(klog.FromContext(ctx), "service-ca-template-mapper")
	logger.WithValues("podNamespace", pod.Namespace, "podName", pod.Name)
	ctx = klog.NewContext(ctx, logger)

	csr, err := x509.ParseCertificateRequest(pcr.Spec.StubPKCS10Request)
	if err != nil {
		return nil, fmt.Errorf("failed to parse CSR: %w", err)
	}

	serial, err := serialGen()
	if err != nil {
		return nil, fmt.Errorf("failed to generate serial number: %w", err)
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

	dnsNames := sets.New[string]()

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

		svc, err := func() (svc *corev1.Service, err error) {
			m.missingServicesLock.Lock() // TODO: could use striped locking
			defer m.missingServicesLock.Unlock()

			defer func() {
				if svc == nil && err == nil {
					key := types.NamespacedName{Namespace: pcr.Namespace, Name: subdomain}
					m.missingServices[key] = append(m.missingServices[key], types.NamespacedName{Namespace: pcr.Namespace, Name: pcr.Name})
				}
			}()

			svc, err = liveFallbackGet(ctx, m.ServiceLister, m.ServiceClient, pcr.Namespace, subdomain)
			if errors.IsNotFound(err) {
				logger.Error(err, "service for the pod's subdomain not found", "subdomain", subdomain)
				return nil, nil
			} else if err != nil {
				return nil, err
			}

			if !(svc.Spec.Type == corev1.ServiceTypeClusterIP && svc.Spec.ClusterIP == "None") {
				logger.Info("service is not a headless service", "serviceName", svc.Name)
				return nil, nil
			}

			if !selectorMatchesLabels(svc.Spec.Selector, pod.Labels) {
				logger.Info("pod does not match service's selector", "serviceName", svc.Name)
				return nil, nil
			}

			return svc, nil
		}()
		if svc == nil {
			return nil, err
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
			svc, err := func() (svc *corev1.Service, err error) {
				m.missingServicesLock.Lock() // TODO: could use striped locking
				defer m.missingServicesLock.Unlock()

				defer func() {
					if svc == nil && err == nil {
						key := types.NamespacedName{Namespace: pcr.Namespace, Name: svcName}
						m.missingServices[key] = append(m.missingServices[key], types.NamespacedName{Namespace: pcr.Namespace, Name: pcr.Name})
					}
				}()

				svc, err = liveFallbackGet(ctx, m.ServiceLister, m.ServiceClient, pcr.Namespace, svcName)
				if errors.IsNotFound(err) {
					logger.Error(err, "service for the pod not found", "svcName", svcName)
					return nil, nil
				} else if err != nil {
					return nil, err
				}

				if !selectorMatchesLabels(svc.Spec.Selector, pod.Labels) {
					logger.Info("pod does not match service's selector", "serviceName", svc.Name)
					return nil, nil
				}

				return svc, nil
			}()
			if svc == nil {
				return nil, err
			}
			dnsNames.Insert(strings.Join([]string{svc.Name, svc.Namespace, "svc", m.ClusterDomain}, "."))
		}
	}

	if len(dnsNames) == 0 {
		return nil, fmt.Errorf("the config from UnverifiedUserAnnotations would end up in a certificate without any SubjectAlternativeNames, thus being unusable for TLS serving")
	}
	template.DNSNames = sets.List(dnsNames)

	return template, nil
}

// liveFallbackGet attempts to fetch a service in cache, and then in client if cache lookup fails.
//
// WARNING: The returned service can be a pointer to the local cache and is not safe to be written into!
func liveFallbackGet(ctx context.Context, lister corev1listers.ServiceLister, client corev1clients.ServicesGetter, svcNamespace, svcName string) (*corev1.Service, error) {
	svc, err := lister.Services(svcNamespace).Get(svcName)
	if err == nil {
		return svc, err
	}
	return client.Services(svcNamespace).Get(ctx, svcName, metav1.GetOptions{})

}

func isPCRInFinalState(pcr *certv1.PodCertificateRequestStatus) bool {
	for _, c := range pcr.Conditions {
		switch c.Type {
		case certv1.PodCertificateRequestConditionTypeIssued,
			certv1.PodCertificateRequestConditionTypeDenied,
			certv1.PodCertificateRequestConditionTypeFailed:
			if c.Status == metav1.ConditionTrue {
				return true
			}
		}
	}
	return false
}

func selectorMatchesLabels(selector, toMatch map[string]string) bool {
	if len(selector) == 0 {
		return false
	}
	matcher := labels.Set(selector).AsSelectorPreValidated()
	return matcher.Matches(labels.Set(toMatch))
}

func serialGen() (*big.Int, error) {
	// returns a uniform random value in [0, max-1), then add 1 to serial to make it a uniform random value in [1, max).
	serial, err := rand.Int(rand.Reader, new(big.Int).SetInt64(math.MaxInt64-1))
	if err != nil {
		return nil, err
	}
	return new(big.Int).Add(serial, big.NewInt(1)), nil
}
