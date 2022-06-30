/*
Copyright 2021 The Kubernetes Authors.

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

package certificates

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509/pkix"
	"encoding/pem"
	"os"
	"path"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	certificatesv1 "k8s.io/api/certificates/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured/unstructuredscheme"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/client-go/util/certificate/csr"
	"k8s.io/client-go/util/keyutil"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/certificates/signer"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestCSRDuration(t *testing.T) {
	t.Parallel()

	s := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	t.Cleanup(s.TearDownFn)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	t.Cleanup(cancel)

	// assert that the metrics we collect during the test run match expectations
	// we have 7 valid test cases below that request a duration of which 6 should have their duration honored
	wantMetricStrings := []string{
		`apiserver_certificates_registry_csr_honored_duration_total{signerName="kubernetes.io/kube-apiserver-client"} 6`,
		`apiserver_certificates_registry_csr_requested_duration_total{signerName="kubernetes.io/kube-apiserver-client"} 7`,
	}
	t.Cleanup(func() {
		copyConfig := rest.CopyConfig(s.ClientConfig)
		copyConfig.GroupVersion = &schema.GroupVersion{}
		copyConfig.NegotiatedSerializer = unstructuredscheme.NewUnstructuredNegotiatedSerializer()
		rc, err := rest.RESTClientFor(copyConfig)
		if err != nil {
			t.Fatal(err)
		}
		body, err := rc.Get().AbsPath("/metrics").DoRaw(ctx)
		if err != nil {
			t.Fatal(err)
		}
		var gotMetricStrings []string
		for _, line := range strings.Split(string(body), "\n") {
			if strings.HasPrefix(line, "apiserver_certificates_registry_") {
				gotMetricStrings = append(gotMetricStrings, line)
			}
		}
		if diff := cmp.Diff(wantMetricStrings, gotMetricStrings); diff != "" {
			t.Errorf("unexpected metrics diff (-want +got): %s", diff)
		}
	})

	client := clientset.NewForConfigOrDie(s.ClientConfig)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	caPrivateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	caCert, err := certutil.NewSelfSignedCACert(certutil.Config{CommonName: "test-ca"}, caPrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	caPublicKeyFile := path.Join(s.TmpDir, "test-ca-public-key")
	if err := os.WriteFile(caPublicKeyFile, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caCert.Raw}), os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}
	caPrivateKeyBytes, err := keyutil.MarshalPrivateKeyToPEM(caPrivateKey)
	if err != nil {
		t.Fatal(err)
	}
	caPrivateKeyFile := path.Join(s.TmpDir, "test-ca-private-key")
	if err := os.WriteFile(caPrivateKeyFile, caPrivateKeyBytes, os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	c, err := signer.NewKubeAPIServerClientCSRSigningController(client, informerFactory.Certificates().V1().CertificateSigningRequests(), caPublicKeyFile, caPrivateKeyFile, 24*time.Hour)
	if err != nil {
		t.Fatal(err)
	}

	informerFactory.Start(ctx.Done())
	go c.Run(ctx, 1)

	tests := []struct {
		name, csrName          string
		duration, wantDuration time.Duration
		wantError              string
	}{
		{
			name:         "no duration set",
			duration:     0,
			wantDuration: 24 * time.Hour,
			wantError:    "",
		},
		{
			name:         "same duration set as certTTL",
			duration:     24 * time.Hour,
			wantDuration: 24 * time.Hour,
			wantError:    "",
		},
		{
			name:         "longer duration than certTTL",
			duration:     48 * time.Hour,
			wantDuration: 24 * time.Hour,
			wantError:    "",
		},
		{
			name:         "slightly shorter duration set",
			duration:     20 * time.Hour,
			wantDuration: 20 * time.Hour,
			wantError:    "",
		},
		{
			name:         "even shorter duration set",
			duration:     10 * time.Hour,
			wantDuration: 10 * time.Hour,
			wantError:    "",
		},
		{
			name:         "short duration set",
			duration:     2 * time.Hour,
			wantDuration: 2*time.Hour + 5*time.Minute,
			wantError:    "",
		},
		{
			name:         "very short duration set",
			duration:     30 * time.Minute,
			wantDuration: 30*time.Minute + 5*time.Minute,
			wantError:    "",
		},
		{
			name:         "shortest duration set",
			duration:     10 * time.Minute,
			wantDuration: 10*time.Minute + 5*time.Minute,
			wantError:    "",
		},
		{
			name:         "just too short duration set",
			csrName:      "invalid-csr-001",
			duration:     10*time.Minute - time.Second,
			wantDuration: 0,
			wantError: `cannot create certificate signing request: ` +
				`CertificateSigningRequest.certificates.k8s.io "invalid-csr-001" is invalid: spec.expirationSeconds: Invalid value: 599: may not specify a duration less than 600 seconds (10 minutes)`,
		},
		{
			name:         "really too short duration set",
			csrName:      "invalid-csr-002",
			duration:     3 * time.Minute,
			wantDuration: 0,
			wantError: `cannot create certificate signing request: ` +
				`CertificateSigningRequest.certificates.k8s.io "invalid-csr-002" is invalid: spec.expirationSeconds: Invalid value: 180: may not specify a duration less than 600 seconds (10 minutes)`,
		},
		{
			name:         "negative duration set",
			csrName:      "invalid-csr-003",
			duration:     -7 * time.Minute,
			wantDuration: 0,
			wantError: `cannot create certificate signing request: ` +
				`CertificateSigningRequest.certificates.k8s.io "invalid-csr-003" is invalid: spec.expirationSeconds: Invalid value: -420: may not specify a duration less than 600 seconds (10 minutes)`,
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
			if err != nil {
				t.Fatal(err)
			}
			csrData, err := certutil.MakeCSR(privateKey, &pkix.Name{CommonName: "panda"}, nil, nil)
			if err != nil {
				t.Fatal(err)
			}

			csrName, csrUID, errReq := csr.RequestCertificate(client, csrData, tt.csrName, certificatesv1.KubeAPIServerClientSignerName,
				durationPtr(tt.duration), []certificatesv1.KeyUsage{certificatesv1.UsageClientAuth}, privateKey)

			if diff := cmp.Diff(tt.wantError, errStr(errReq)); len(diff) > 0 {
				t.Fatalf("CSR input duration %v err diff (-want, +got):\n%s", tt.duration, diff)
			}

			if len(tt.wantError) > 0 {
				return
			}

			csrObj, err := client.CertificatesV1().CertificateSigningRequests().Get(ctx, csrName, metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			csrObj.Status.Conditions = []certificatesv1.CertificateSigningRequestCondition{
				{
					Type:    certificatesv1.CertificateApproved,
					Status:  v1.ConditionTrue,
					Reason:  "TestCSRDuration",
					Message: t.Name(),
				},
			}
			_, err = client.CertificatesV1().CertificateSigningRequests().UpdateApproval(ctx, csrName, csrObj, metav1.UpdateOptions{})
			if err != nil {
				t.Fatal(err)
			}

			certData, err := csr.WaitForCertificate(ctx, client, csrName, csrUID)
			if err != nil {
				t.Fatal(err)
			}

			certs, err := certutil.ParseCertsPEM(certData)
			if err != nil {
				t.Fatal(err)
			}

			switch l := len(certs); l {
			case 1:
				// good
			default:
				t.Errorf("expected 1 cert, got %d", l)
				for i, certificate := range certs {
					t.Log(i, dynamiccertificates.GetHumanCertDetail(certificate))
				}
				t.FailNow()
			}

			cert := certs[0]

			if got := cert.NotAfter.Sub(cert.NotBefore); got != tt.wantDuration {
				t.Errorf("CSR input duration %v got duration = %v, want %v\n%s", tt.duration, got, tt.wantDuration, dynamiccertificates.GetHumanCertDetail(cert))
			}
		})
	}
}

func durationPtr(duration time.Duration) *time.Duration {
	if duration == 0 {
		return nil
	}
	return &duration
}

func errStr(err error) string {
	if err == nil {
		return ""
	}
	es := err.Error()
	if len(es) == 0 {
		panic("invalid empty error")
	}
	return es
}
