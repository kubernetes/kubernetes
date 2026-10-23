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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	certv1 "k8s.io/api/certificates/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	corev1listers "k8s.io/client-go/listers/core/v1"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/keyutil"
	"k8s.io/client-go/util/workqueue"
)

const testSignerName = "test.stuff/testing-signer"

func TestSelectorMatchesLabels(t *testing.T) {
	notMatched := &SigningErrorEvent{
		Reason:  serviceLabelsNotMatchedReason,
		Message: `pod does not match "my-svc" service's selector`,
	}

	tests := []struct {
		name        string
		selector    map[string]string
		podLabels   map[string]string
		expectedEvt *SigningErrorEvent
	}{
		{
			name:      "selector matches pod labels exactly",
			selector:  map[string]string{"app": "foo"},
			podLabels: map[string]string{"app": "foo"},
		},
		{
			name:      "selector is a subset of pod labels",
			selector:  map[string]string{"app": "foo"},
			podLabels: map[string]string{"app": "foo", "tier": "backend"},
		},
		{
			name:      "multi-key selector fully matches",
			selector:  map[string]string{"app": "foo", "tier": "backend"},
			podLabels: map[string]string{"app": "foo", "tier": "backend", "extra": "x"},
		},
		{
			name:        "label value differs",
			selector:    map[string]string{"app": "foo"},
			podLabels:   map[string]string{"app": "bar"},
			expectedEvt: notMatched,
		},
		{
			name:        "selector key missing from pod",
			selector:    map[string]string{"app": "foo"},
			podLabels:   map[string]string{"tier": "backend"},
			expectedEvt: notMatched,
		},
		{
			name:        "only part of a multi-key selector matches",
			selector:    map[string]string{"app": "foo", "tier": "backend"},
			podLabels:   map[string]string{"app": "foo"},
			expectedEvt: notMatched,
		},
		{
			name:        "pod without labels",
			selector:    map[string]string{"app": "foo"},
			podLabels:   nil,
			expectedEvt: notMatched,
		},
		{
			name:        "nil selector never matches",
			selector:    nil,
			podLabels:   map[string]string{"app": "foo"},
			expectedEvt: notMatched,
		},
		{
			name:        "empty selector never matches",
			selector:    map[string]string{},
			podLabels:   nil,
			expectedEvt: notMatched,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := &corev1.Service{
				Name: "my-svc", Namespace: "ns",
				Spec: corev1.ServiceSpec{Selector: tc.selector},
			}
			pod := &corev1.Pod{
				Name: "my-pod", Namespace: "ns", Labels: tc.podLabels,
			}

			got := selectorMatchesLabels(svc, pod)
			if diff := cmp.Diff(tc.expectedEvt, got); diff != "" {
				t.Errorf("unexpected event (-want +got):\n%s", diff)
			}
		})
	}
}

func TestKubeServiceCATemplateMapper_MapPodCertRequestToX509Template(t *testing.T) {
	testNamespace := "test-ns"

	tests := []struct {
		name             string
		skipServiceCache bool
		pcr              *certv1.PodCertificateRequest
		pod              *corev1.Pod
		svc              *corev1.Service
		wantCert         *x509.Certificate
		wantLifetime     time.Duration // zero means the default of 1 hour set by newPCRBuilder
		wantEvent        *SigningErrorEvent
		wantErr          bool
	}{
		{
			name: "pod requesting an existing service",
			pcr:  newPCRBuilder(t, testNamespace).WithServices("my-svc").B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Labels: map[string]string{"app": "foo"},
			},
			svc: &corev1.Service{
				Name: "my-svc", Namespace: testNamespace,
				Spec: corev1.ServiceSpec{Selector: map[string]string{"app": "foo"}},
			},
			wantCert: &x509.Certificate{
				DNSNames: []string{"my-svc.test-ns.svc.nondefault.domain"},
				KeyUsage: x509.KeyUsageDigitalSignature,
			},
		},
		{
			name: "pod requesting a service absent from the cache",
			pcr:  newPCRBuilder(t, testNamespace).WithServices("my-svc").B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Labels: map[string]string{"app": "foo"},
			},
			skipServiceCache: true,
			svc: &corev1.Service{
				Name: "my-svc", Namespace: testNamespace,
				Spec: corev1.ServiceSpec{Selector: map[string]string{"app": "foo"}},
			},
			wantCert: &x509.Certificate{
				DNSNames: []string{"my-svc.test-ns.svc.nondefault.domain"},
				KeyUsage: x509.KeyUsageDigitalSignature,
			},
		},
		{
			name: "pod requesting a nonexistent service",
			pcr:  newPCRBuilder(t, testNamespace).WithServices("my-svc").B(),
			pod:  &corev1.Pod{Name: "my-pod", Namespace: testNamespace},
			wantEvent: &SigningErrorEvent{
				Reason:  refServiceNotFoundReason,
				Message: `service "my-svc" for the pod not found`,
			},
		},
		{
			name: "pod requesting a service in another namespace",
			pcr:  newPCRBuilder(t, testNamespace).WithServices("my-svc").B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Labels: map[string]string{"app": "foo"},
			},
			svc: &corev1.Service{
				Name: "my-svc", Namespace: "other-ns",
				Spec: corev1.ServiceSpec{Selector: map[string]string{"app": "foo"}},
			},
			wantEvent: &SigningErrorEvent{
				Reason:  refServiceNotFoundReason,
				Message: `service "my-svc" for the pod not found`,
			},
		},
		{
			name: "pod not matching the service selector",
			pcr:  newPCRBuilder(t, testNamespace).WithServices("my-svc").B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Labels: map[string]string{"app": "bar"},
			},
			svc: &corev1.Service{
				Name: "my-svc", Namespace: testNamespace,
				Spec: corev1.ServiceSpec{Selector: map[string]string{"app": "foo"}},
			},
			wantEvent: &SigningErrorEvent{
				Reason:  serviceLabelsNotMatchedReason,
				Message: `pod does not match "my-svc" service's selector`,
			},
		},
		{
			name: "pod requesting a service without a selector",
			pcr:  newPCRBuilder(t, testNamespace).WithServices("my-svc").B(),
			pod:  &corev1.Pod{Name: "my-pod", Namespace: testNamespace},
			svc:  &corev1.Service{Name: "my-svc", Namespace: testNamespace},
			wantEvent: &SigningErrorEvent{
				Reason:  serviceLabelsNotMatchedReason,
				Message: `pod does not match "my-svc" service's selector`,
			},
		},
		{
			name: "pod requesting the same service multiple times",
			pcr:  newPCRBuilder(t, testNamespace).WithServices("my-svc", "my-svc").B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Labels: map[string]string{"app": "foo"},
			},
			svc: &corev1.Service{
				Name: "my-svc", Namespace: testNamespace,
				Spec: corev1.ServiceSpec{Selector: map[string]string{"app": "foo"}},
			},
			wantCert: &x509.Certificate{
				DNSNames: []string{"my-svc.test-ns.svc.nondefault.domain"},
				KeyUsage: x509.KeyUsageDigitalSignature,
			},
		},
		{
			name: "pod requesting an existing and a nonexistent service",
			pcr:  newPCRBuilder(t, testNamespace).WithServices("my-svc", "missing-svc").B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Labels: map[string]string{"app": "foo"},
			},
			svc: &corev1.Service{
				Name: "my-svc", Namespace: testNamespace,
				Spec: corev1.ServiceSpec{Selector: map[string]string{"app": "foo"}},
			},
			wantEvent: &SigningErrorEvent{
				Reason:  refServiceNotFoundReason,
				Message: `service "missing-svc" for the pod not found`,
			},
		},
		{
			name: "pod requesting its FQDN using its name",
			pcr:  newPCRBuilder(t, testNamespace).WithFQDN().B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Spec: corev1.PodSpec{Subdomain: "my-subdomain"},
			},
			wantCert: &x509.Certificate{
				DNSNames: []string{"my-pod.my-subdomain.test-ns.svc.nondefault.domain"},
				KeyUsage: x509.KeyUsageDigitalSignature,
			},
		},
		{
			name: "pod requesting its FQDN using an explicit hostname",
			pcr:  newPCRBuilder(t, testNamespace).WithFQDN().B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Spec: corev1.PodSpec{Hostname: "my-host", Subdomain: "my-subdomain"},
			},
			wantCert: &x509.Certificate{
				DNSNames: []string{"my-host.my-subdomain.test-ns.svc.nondefault.domain"},
				KeyUsage: x509.KeyUsageDigitalSignature,
			},
		},
		{
			name: "pod requesting both its FQDN and a service",
			pcr:  newPCRBuilder(t, testNamespace).WithFQDN().WithServices("my-svc").B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Labels: map[string]string{"app": "foo"},
				Spec:   corev1.PodSpec{Subdomain: "my-subdomain"},
			},
			svc: &corev1.Service{
				Name: "my-svc", Namespace: testNamespace,
				Spec: corev1.ServiceSpec{Selector: map[string]string{"app": "foo"}},
			},
			wantCert: &x509.Certificate{
				DNSNames: []string{
					"my-pod.my-subdomain.test-ns.svc.nondefault.domain",
					"my-svc.test-ns.svc.nondefault.domain",
				},
				KeyUsage: x509.KeyUsageDigitalSignature,
			},
		},
		{
			name:    "pod requesting its FQDN without a subdomain",
			pcr:     newPCRBuilder(t, testNamespace).WithFQDN().B(),
			pod:     &corev1.Pod{Name: "my-pod", Namespace: testNamespace},
			wantErr: true,
		},
		{
			name:    "pod requesting neither its FQDN nor services",
			pcr:     newPCRBuilder(t, testNamespace).B(),
			pod:     &corev1.Pod{Name: "my-pod", Namespace: testNamespace},
			wantErr: true,
		},
		{
			name: "requested expiration below the signer maximum is honored",
			pcr:  newPCRBuilder(t, testNamespace).WithFQDN().WithRequestMaxExpiration(1800).B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Spec: corev1.PodSpec{Subdomain: "my-subdomain"},
			},
			wantCert: &x509.Certificate{
				DNSNames: []string{"my-pod.my-subdomain.test-ns.svc.nondefault.domain"},
				KeyUsage: x509.KeyUsageDigitalSignature,
			},
			wantLifetime: 30 * time.Minute,
		},
		{
			name: "requested expiration above the signer maximum is capped",
			pcr:  newPCRBuilder(t, testNamespace).WithFQDN().WithRequestMaxExpiration(30 * 24 * 60 * 60).B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Spec: corev1.PodSpec{Subdomain: "my-subdomain"},
			},
			wantCert: &x509.Certificate{
				DNSNames: []string{"my-pod.my-subdomain.test-ns.svc.nondefault.domain"},
				KeyUsage: x509.KeyUsageDigitalSignature,
			},
			wantLifetime: signerMaxExpirationSeconds * time.Second,
		},
		{
			name: "RSA key request adds key encipherment usage",
			pcr:  newPCRBuilder(t, testNamespace).WithFQDN().WithCSR(rsaCSR(t)).B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Spec: corev1.PodSpec{Subdomain: "my-subdomain"},
			},
			wantCert: &x509.Certificate{
				DNSNames: []string{"my-pod.my-subdomain.test-ns.svc.nondefault.domain"},
				KeyUsage: x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
			},
		},
		{
			name: "malformed stub PKCS#10 request",
			pcr:  newPCRBuilder(t, testNamespace).WithFQDN().WithCSR([]byte("not a csr")).B(),
			pod: &corev1.Pod{
				Name: "my-pod", Namespace: testNamespace,
				Spec: corev1.PodSpec{Subdomain: "my-subdomain"},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			objs := []runtime.Object{}
			if tt.svc != nil {
				objs = append(objs, tt.svc)
			}
			fakeClient := fake.NewClientset(objs...)
			listerIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
			if tt.svc != nil && !tt.skipServiceCache {
				_ = listerIndexer.Add(tt.svc)
			}
			svcLister := corev1listers.NewServiceLister(listerIndexer)

			m := NewKubeServiceCATemplateMapper(fakeClient.CoreV1(), &alwaysSyncedInformer{}, svcLister, "nondefault.domain")
			gotTmpl, gotEvent, gotErr := m.MapPodCertRequestToX509Template(t.Context(), tt.pcr, tt.pod)
			if gotErr != nil {
				if !tt.wantErr {
					t.Errorf("MapPodCertRequestToX509Template() failed unexpectedly: %v", gotErr)
				}
				return
			}
			if tt.wantErr {
				t.Fatal("MapPodCertRequestToX509Template() succeeded unexpectedly")
			}
			if diff := cmp.Diff(tt.wantEvent, gotEvent); diff != "" {
				t.Errorf("unexpected event (-want +got):\n%s", diff)
			}

			if tt.wantCert == nil {
				if gotTmpl != nil {
					t.Errorf("expected no certificate template, got %+v", gotTmpl)
				}
				return
			}

			if gotTmpl == nil {
				t.Fatal("expected a certificate template, got nil")
			}

			// Serial number, validity and public key are generated per call, so they're not compared.
			if diff := cmp.Diff(tt.wantCert.DNSNames, gotTmpl.DNSNames); diff != "" {
				t.Errorf("unexpected DNSNames (-want +got):\n%s", diff)
			}
			if tt.wantCert.KeyUsage != gotTmpl.KeyUsage {
				t.Errorf("expected KeyUsage %v, got %v", tt.wantCert.KeyUsage, gotTmpl.KeyUsage)
			}
			if diff := cmp.Diff([]x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}, gotTmpl.ExtKeyUsage); diff != "" {
				t.Errorf("unexpected ExtKeyUsage (-want +got):\n%s", diff)
			}
			wantLifetime := tt.wantLifetime
			if wantLifetime == 0 {
				wantLifetime = time.Hour
			}
			if got := gotTmpl.NotAfter.Sub(gotTmpl.NotBefore); got != wantLifetime {
				t.Errorf("expected certificate lifetime %v, got %v", wantLifetime, got)
			}
		})
	}
}

func TestPodCertsSignerController_syncPodCertificateRequest(t *testing.T) {
	const testNamespace = "test-namespace"

	certPath, keyPath := genTmpCertKey(t)

	// Any mapper call in a case that expects the sync to bail out early surfaces as an error.
	unexpectedMapperCall := &testCertMapper{err: errors.New("mapper must not be called")}

	validTmpl, err := templateFor(eccCSR(t), 3600)
	if err != nil {
		t.Fatalf("failed to create a certificate template: %v", err)
	}
	validTmpl.DNSNames = []string{"my-svc.test-namespace.svc.cluster.local"}

	shortLivedTmpl := *validTmpl
	shortLivedTmpl.NotAfter = shortLivedTmpl.NotBefore.Add(10 * time.Minute)
	longLivedTmpl := *validTmpl
	longLivedTmpl.NotAfter = longLivedTmpl.NotBefore.Add(14 * 24 * time.Hour)

	testPod := &corev1.Pod{Name: "test-pod", Namespace: testNamespace, UID: "test-uid"}
	podPCR := func() *pcrBuilder { return newPCRBuilder(t, testNamespace).WithPod("test-pod", "test-uid") }

	tests := []struct {
		name               string
		certTemplateMapper X509TemplateMapper
		pod                *corev1.Pod
		pcr                *certv1.PodCertificateRequest
		statusUpdateErr    error  // returned by the client on PCR status updates
		wantErr            string // substring of the expected error, empty means success
		wantEventReason    string
		wantIssued         bool
	}{
		{
			name:               "PCR is already gone",
			certTemplateMapper: unexpectedMapperCall,
		},
		{
			name:               "PCR exists but the pod's gone",
			certTemplateMapper: unexpectedMapperCall,
			pcr:                podPCR().B(),
			wantErr:            "failed to find a pod",
		},
		{
			name:               "PCR is being deleted",
			certTemplateMapper: unexpectedMapperCall,
			pod:                testPod,
			pcr:                podPCR().WithDeletionTimestamp().B(),
		},
		{
			name:               "PCR in final state: issued",
			certTemplateMapper: unexpectedMapperCall,
			pod:                testPod,
			pcr:                podPCR().WithCondition(certv1.PodCertificateRequestConditionTypeIssued, metav1.ConditionTrue).B(),
		},
		{
			name:               "PCR in final state: denied",
			certTemplateMapper: unexpectedMapperCall,
			pod:                testPod,
			pcr:                podPCR().WithCondition(certv1.PodCertificateRequestConditionTypeDenied, metav1.ConditionTrue).B(),
		},
		{
			name:               "PCR in final state: failed",
			certTemplateMapper: unexpectedMapperCall,
			pod:                testPod,
			pcr:                podPCR().WithCondition(certv1.PodCertificateRequestConditionTypeFailed, metav1.ConditionTrue).B(),
		},
		{
			name:               "PCR with a false final condition is still processed",
			certTemplateMapper: &testCertMapper{tmpl: validTmpl},
			pod:                testPod,
			pcr:                podPCR().WithCondition(certv1.PodCertificateRequestConditionTypeIssued, metav1.ConditionFalse).B(),
			wantIssued:         true,
		},
		{
			name:               "PCR for a different signer",
			certTemplateMapper: unexpectedMapperCall,
			pod:                testPod,
			pcr:                podPCR().WithSigner("other.stuff/other-signer").B(),
			wantErr:            "signer name doesn't match",
		},
		{
			name:               "pod UID differs from the one in PCR",
			certTemplateMapper: unexpectedMapperCall,
			pod:                &corev1.Pod{Name: "test-pod", Namespace: testNamespace, UID: "recreated-uid"},
			pcr:                podPCR().B(),
			wantErr:            "pod UIDs don't match",
		},
		{
			name:               "mapper fails",
			certTemplateMapper: &testCertMapper{err: errors.New("mapping went sideways")},
			pod:                testPod,
			pcr:                podPCR().B(),
			wantErr:            "mapping went sideways",
		},
		{
			name:               "mapper reports an event and no template",
			certTemplateMapper: &testCertMapper{ev: &SigningErrorEvent{Reason: refServiceNotFoundReason, Message: "no such service"}},
			pod:                testPod,
			pcr:                podPCR().B(),
			wantEventReason:    refServiceNotFoundReason,
		},
		{
			name:               "mapper returns neither a template nor an event",
			certTemplateMapper: &testCertMapper{},
			pod:                testPod,
			pcr:                podPCR().B(),
		},
		{
			name:               "template cannot be signed",
			certTemplateMapper: &testCertMapper{tmpl: &x509.Certificate{SerialNumber: validTmpl.SerialNumber}},
			pod:                testPod,
			pcr:                podPCR().B(),
			wantErr:            "failed to sign cert",
		},
		{
			name:               "valid request is issued",
			certTemplateMapper: &testCertMapper{tmpl: validTmpl},
			pod:                testPod,
			pcr:                podPCR().B(),
			wantIssued:         true,
		},
		{
			name:               "short-lived certificate is issued",
			certTemplateMapper: &testCertMapper{tmpl: &shortLivedTmpl},
			pod:                testPod,
			pcr:                podPCR().B(),
			wantIssued:         true,
		},
		{
			name:               "long-lived certificate is issued",
			certTemplateMapper: &testCertMapper{tmpl: &longLivedTmpl},
			pod:                testPod,
			pcr:                podPCR().B(),
			wantIssued:         true,
		},
		{
			name:               "status update fails",
			certTemplateMapper: &testCertMapper{tmpl: validTmpl},
			pod:                testPod,
			pcr:                podPCR().B(),
			statusUpdateErr:    errors.New("oh noes"),
			wantErr:            "oh noes",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objs []runtime.Object
			if tt.pcr != nil {
				objs = append(objs, tt.pcr)
			}
			if tt.pod != nil {
				objs = append(objs, tt.pod)
			}
			fakeClient := fake.NewClientset(objs...)

			var gotUpdated *certv1.PodCertificateRequest
			fakeClient.PrependReactor("update", "podcertificaterequests", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if tt.statusUpdateErr != nil {
					return true, nil, tt.statusUpdateErr
				}
				gotUpdated = action.(clienttesting.UpdateAction).GetObject().(*certv1.PodCertificateRequest)
				return true, gotUpdated, nil
			})

			podInformer := informers.NewSharedInformerFactory(fakeClient, 0).Core().V1().Pods()
			if tt.pod != nil {
				if err := podInformer.Informer().GetIndexer().Add(tt.pod); err != nil {
					t.Fatalf("failed to add pod to the informer cache: %v", err)
				}
			}

			s, err := NewPodCertificatesSigner(t.Context(), testSignerName, certPath, keyPath, tt.certTemplateMapper, fakeClient, podInformer)
			if err != nil {
				t.Fatalf("could not construct receiver type: %v", err)
			}

			// The controller sets up its own PCR informer, we need to populate it
			if tt.pcr != nil {
				if err := s.podCertRequestsInformer.GetIndexer().Add(tt.pcr); err != nil {
					t.Fatalf("failed to add PodCertificateRequest to the informer cache: %v", err)
				}
			}

			recorder := record.NewFakeRecorder(10)
			s.eventRecorder = recorder

			gotErr := s.syncPodCertificateRequest(t.Context(), types.NamespacedName{Namespace: testNamespace, Name: "pcr-req"})
			switch {
			case gotErr != nil && tt.wantErr == "":
				t.Fatalf("syncPodCertificateRequest() failed unexpectedly: %v", gotErr)
			case gotErr != nil && !strings.Contains(gotErr.Error(), tt.wantErr):
				t.Fatalf("expected an error containing %q, got: %v", tt.wantErr, gotErr)
			case gotErr == nil && tt.wantErr != "":
				t.Fatalf("syncPodCertificateRequest() succeeded, expected an error containing %q", tt.wantErr)
			}

			var gotEvent string
			select {
			case gotEvent = <-recorder.Events:
			default:
			}
			if tt.wantEventReason == "" && gotEvent != "" {
				t.Errorf("unexpected event: %q", gotEvent)
			}
			if tt.wantEventReason != "" && !strings.Contains(gotEvent, tt.wantEventReason) {
				t.Errorf("expected an event with reason %q, got %q", tt.wantEventReason, gotEvent)
			}

			if !tt.wantIssued {
				if gotUpdated != nil {
					t.Errorf("expected no status update, got %+v", gotUpdated.Status)
				}
				return
			}
			if gotUpdated == nil {
				t.Fatal("expected the PodCertificateRequest status to be updated")
			}

			chain, err := certutil.ParseCertsPEM([]byte(gotUpdated.Status.CertificateChain))
			if err != nil {
				t.Fatalf("failed to parse the issued certificate chain: %v", err)
			}
			if len(chain) != 2 {
				t.Fatalf("expected the leaf and the CA in the chain, got %d certificates", len(chain))
			}
			if err := chain[0].CheckSignatureFrom(chain[1]); err != nil {
				t.Errorf("leaf is not signed by the CA from the chain: %v", err)
			}
			if diff := cmp.Diff(validTmpl.DNSNames, chain[0].DNSNames); diff != "" {
				t.Errorf("unexpected DNSNames (-want +got):\n%s", diff)
			}
			if gotUpdated.Status.NotBefore == nil || gotUpdated.Status.NotAfter == nil || gotUpdated.Status.BeginRefreshAt == nil {
				t.Fatalf("expected validity and refresh timestamps to be set, got %+v", gotUpdated.Status)
			}
			lifetime := gotUpdated.Status.NotAfter.Sub(gotUpdated.Status.NotBefore.Time)
			refreshDelay := gotUpdated.Status.BeginRefreshAt.Sub(gotUpdated.Status.NotBefore.Time)
			if want := lifetime * 3 / 5; refreshDelay != want {
				t.Errorf("expected refresh at 60%% of certificate lifetime (%v), got %v", want, refreshDelay)
			}
			if len(gotUpdated.Status.Conditions) != 1 ||
				gotUpdated.Status.Conditions[0].Type != certv1.PodCertificateRequestConditionTypeIssued ||
				gotUpdated.Status.Conditions[0].Status != metav1.ConditionTrue {
				t.Errorf("expected a single true Issued condition, got %+v", gotUpdated.Status.Conditions)
			}
		})
	}
}

var _ X509TemplateMapper = &testCertMapper{}

type testCertMapper struct {
	err  error
	ev   *SigningErrorEvent
	tmpl *x509.Certificate
}

func (m *testCertMapper) HookEvents(q workqueue.TypedRateLimitingInterface[types.NamespacedName]) {}
func (m *testCertMapper) MapPodCertRequestToX509Template(_ context.Context, pcr *certv1.PodCertificateRequest, pod *corev1.Pod) (*x509.Certificate, *SigningErrorEvent, error) {
	return m.tmpl, m.ev, m.err
}

func genTmpCertKey(t *testing.T) (certPath, keyPath string) {
	t.Helper()

	tmpDir := t.TempDir()
	certPath = filepath.Join(tmpDir, "tls.crt")
	keyPath = filepath.Join(tmpDir, "tls.key")

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("failed to generate private key: %v", err)
	}
	cert, err := certutil.NewSelfSignedCACert(certutil.Config{CommonName: "pod-certs-signer-test-ca"}, key)
	if err != nil {
		t.Fatalf("failed to create CA certificate: %v", err)
	}
	keyPEM, err := keyutil.MarshalPrivateKeyToPEM(key)
	if err != nil {
		t.Fatalf("failed to PEM-encode the private key: %v", err)
	}
	certPEM, err := certutil.EncodeCertificates(cert)
	if err != nil {
		t.Fatalf("failed to PEM-encode the certificate: %v", err)
	}
	if err := keyutil.WriteKey(keyPath, keyPEM); err != nil {
		t.Fatalf("failed to write the private key to %q: %v", keyPath, err)
	}
	if err := certutil.WriteCert(certPath, certPEM); err != nil {
		t.Fatalf("failed to write the certificate  %v", err)
	}
	return certPath, keyPath
}

type alwaysSyncedInformer struct{ cache.SharedIndexInformer }

type pcrBuilder struct {
	*certv1.PodCertificateRequest
}

func newPCRBuilder(t *testing.T, namespace string) *pcrBuilder {

	return &pcrBuilder{
		PodCertificateRequest: &certv1.PodCertificateRequest{
			Name: "pcr-req", Namespace: namespace,
			Spec: certv1.PodCertificateRequestSpec{
				SignerName:                testSignerName,
				MaxExpirationSeconds:      new(int32(3600)),
				StubPKCS10Request:         eccCSR(t),
				UnverifiedUserAnnotations: map[string]string{},
			},
		},
	}
}

func (b *pcrBuilder) WithRequestMaxExpiration(seconds int32) *pcrBuilder {
	b.Spec.MaxExpirationSeconds = new(seconds)
	return b
}

func (b *pcrBuilder) WithCSR(csr []byte) *pcrBuilder {
	b.Spec.StubPKCS10Request = csr
	return b
}

func (b *pcrBuilder) WithServices(s ...string) *pcrBuilder {
	if len(s) == 0 {
		return b
	}

	servicesString := strings.Join(s, ",")
	b.Spec.UnverifiedUserAnnotations[ServiceNamesKey] = servicesString
	return b
}

func (b *pcrBuilder) WithFQDN() *pcrBuilder {
	b.Spec.UnverifiedUserAnnotations[AddPodFQDNKey] = "true"
	return b
}

func (b *pcrBuilder) WithSigner(name string) *pcrBuilder {
	b.Spec.SignerName = name
	return b
}

func (b *pcrBuilder) WithDeletionTimestamp() *pcrBuilder {
	b.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	return b
}

func (b *pcrBuilder) WithCondition(condType string, status metav1.ConditionStatus) *pcrBuilder {
	b.Status.Conditions = append(b.Status.Conditions, metav1.Condition{Type: condType, Status: status})
	return b
}

func (b *pcrBuilder) WithPod(podName, podUID string) *pcrBuilder {
	b.Spec.PodName = podName
	b.Spec.PodUID = types.UID(podUID)
	return b
}

func (b *pcrBuilder) B() *certv1.PodCertificateRequest {
	return b.PodCertificateRequest
}

func (*alwaysSyncedInformer) HasSynced() bool {
	return true
}

func eccCSR(t *testing.T) []byte {
	t.Helper()

	key, err := ecdsa.GenerateKey(elliptic.P521(), rand.Reader)
	if err != nil {
		t.Fatalf("failed to generate ECDSA P-521 key: %v", err)
	}
	return signStubCSR(t, key)
}

func rsaCSR(t *testing.T) []byte {
	t.Helper()

	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("failed to generate RSA key: %v", err)
	}
	return signStubCSR(t, key)
}

func signStubCSR(t *testing.T, key crypto.Signer) []byte {
	t.Helper()

	csr, err := x509.CreateCertificateRequest(rand.Reader, &x509.CertificateRequest{}, key)
	if err != nil {
		t.Fatalf("failed to create stub PKCS#10 request: %v", err)
	}
	return csr
}
