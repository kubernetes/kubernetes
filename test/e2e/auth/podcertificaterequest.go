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

package auth

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"math/big"
	"time"

	certificatesv1 "k8s.io/api/certificates/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econformance "k8s.io/kubernetes/test/e2e/framework/conformance"
)

var _ = SIGDescribe("PodCertificateRequest API [Privileged:ClusterAdmin]", func() {
	f := framework.NewDefaultFramework("podcertificaterequest")

	/*
		Release: v1.37
		Testname: PodCertificateRequest API operations
		Description:
		The certificates.k8s.io API group MUST exist in the /apis discovery document.
		The certificates.k8s.io/v1 API group/version MUST exist in the /apis/certificates.k8s.io discovery document.
		The podcertificaterequests and podcertificaterequests/status resources MUST exist in the /apis/certificates.k8s.io/v1 discovery document.
		The podcertificaterequests resource must support namespaced create, read, update, patch, and delete.
		The podcertificaterequests/status resource must support get, update, patch.
	*/
	framework.ConformanceIt("should support PodCertificateRequest API operations", func(ctx context.Context) {
		_, csrData, pemCert, notBefore, notAfter, err := generateKeyCSRAndCert()
		framework.ExpectNoError(err)

		nowTime := metav1.NewTime(time.Now())
		notBeforeTime := metav1.NewTime(notBefore)
		notAfterTime := metav1.NewTime(notAfter)
		beginRefreshTime := metav1.NewTime(notBefore.Add(15 * time.Minute))

		signerName := "example.com/e2e-" + f.UniqueName

		// Construct status patch payloads dynamically because pemCert is dynamic
		statusPatch := map[string]interface{}{
			"status": map[string]interface{}{
				"certificateChain": string(pemCert),
				"notBefore":        notBeforeTime,
				"notAfter":         notAfterTime,
				"beginRefreshAt":   beginRefreshTime,
				"conditions": []map[string]interface{}{
					{
						"type":               certificatesv1.PodCertificateRequestConditionTypeIssued,
						"status":             metav1.ConditionTrue,
						"reason":             "E2E",
						"message":            "Issued by E2E test patch",
						"lastTransitionTime": nowTime,
					},
				},
			},
		}
		statusPatchJSON, err := json.Marshal(statusPatch)
		framework.ExpectNoError(err)
		statusPatchStr := string(statusPatchJSON)

		strategicStatusPatchStr := statusPatchStr

		jsonStatusPatchStr := fmt.Sprintf(`[
			{"op": "add", "path": "/status/certificateChain", "value": %q},
			{"op": "add", "path": "/status/notBefore", "value": %q},
			{"op": "add", "path": "/status/notAfter", "value": %q},
			{"op": "add", "path": "/status/beginRefreshAt", "value": %q},
			{"op": "add", "path": "/status/conditions", "value": [
				{"type": %q, "status": "True", "reason": "E2E", "message": "Issued by E2E test patch", "lastTransitionTime": %q}
			]}
		]`, string(pemCert), notBeforeTime.Format(time.RFC3339), notAfterTime.Format(time.RFC3339), beginRefreshTime.Format(time.RFC3339), certificatesv1.PodCertificateRequestConditionTypeIssued, nowTime.Format(time.RFC3339))

		e2econformance.TestResource(ctx, f,
			&e2econformance.ResourceTestcase[*certificatesv1.PodCertificateRequest]{
				GVR:        certificatesv1.SchemeGroupVersion.WithResource("podcertificaterequests"),
				Namespaced: new(true),
				InitialSpec: &certificatesv1.PodCertificateRequest{
					ObjectMeta: metav1.ObjectMeta{
						GenerateName: "e2e-example-pcr-",
					},
					Spec: certificatesv1.PodCertificateRequestSpec{
						SignerName:           signerName,
						PodName:              "test-pod",
						PodUID:               "12345678-1234-1234-1234-123456789012",
						ServiceAccountName:   "default",
						ServiceAccountUID:    "12345678-1234-1234-1234-123456789013",
						NodeName:             "test-node",
						NodeUID:              "12345678-1234-1234-1234-123456789014",
						MaxExpirationSeconds: new(int32(86400)),
						StubPKCS10Request:    csrData,
					},
				},
				UpdateSpec: func(obj *certificatesv1.PodCertificateRequest) *certificatesv1.PodCertificateRequest {
					if obj.Labels == nil {
						obj.Labels = make(map[string]string)
					}
					obj.Labels["test.podcertificaterequest.example.com"] = "test"
					return obj
				},
				UpdateStatus: func(obj *certificatesv1.PodCertificateRequest) *certificatesv1.PodCertificateRequest {
					obj.Status.CertificateChain = string(pemCert)
					obj.Status.NotBefore = &notBeforeTime
					obj.Status.NotAfter = &notAfterTime
					obj.Status.BeginRefreshAt = &beginRefreshTime
					obj.Status.Conditions = []metav1.Condition{
						{
							Type:               certificatesv1.PodCertificateRequestConditionTypeIssued,
							Status:             metav1.ConditionTrue,
							Reason:             "E2E",
							Message:            "Issued by E2E test",
							LastTransitionTime: nowTime,
						},
					}
					return obj
				},

				ApplyPatchSpec:            `{"metadata": {"labels": {"test.podcertificaterequest.example.com": "test"}}}`,
				StrategicMergePatchSpec:   `{"metadata": {"labels": {"test.podcertificaterequest.example.com": "test"}}}`,
				JSONMergePatchSpec:        `{"metadata": {"labels": {"test.podcertificaterequest.example.com": "test"}}}`,
				JSONPatchSpec:             `[{"op": "add", "path": "/metadata/labels/test.podcertificaterequest.example.com", "value": "test"}]`,
				ApplyPatchStatus:          statusPatchStr,
				StrategicMergePatchStatus: strategicStatusPatchStr,
				JSONMergePatchStatus:      statusPatchStr,
				JSONPatchStatus:           jsonStatusPatchStr,
			},
		)
	})
})

func generateKeyCSRAndCert() (*ecdsa.PrivateKey, []byte, []byte, time.Time, time.Time, error) {
	pk, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, nil, nil, time.Time{}, time.Time{}, err
	}
	tmpl := &x509.CertificateRequest{}
	csrData, err := x509.CreateCertificateRequest(rand.Reader, tmpl, pk)
	if err != nil {
		return nil, nil, nil, time.Time{}, time.Time{}, err
	}
	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		return nil, nil, nil, time.Time{}, time.Time{}, err
	}
	notBefore := time.Now().Add(-1 * time.Minute).Truncate(time.Second) // 1 minute ago, safe within 5 min window
	notAfter := notBefore.Add(time.Hour)
	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			CommonName: "e2e.example.com",
		},
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}
	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &pk.PublicKey, pk)
	if err != nil {
		return nil, nil, nil, time.Time{}, time.Time{}, err
	}
	pemCert := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	return pk, csrData, pemCert, notBefore, notAfter, nil
}
