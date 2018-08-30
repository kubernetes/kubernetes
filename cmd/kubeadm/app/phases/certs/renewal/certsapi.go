/*
Copyright 2018 The Kubernetes Authors.

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

package renewal

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"time"

	"github.com/pkg/errors"

	certsapi "k8s.io/api/certificates/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	certstype "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	certutil "k8s.io/client-go/util/cert"
)

const (
	certAPIPrefixName = "kubeadm-cert"
)

var (
	watchTimeout = 5 * time.Minute
)

// CertsAPIRenewal creates new certificates using the certs API
type CertsAPIRenewal struct {
	client certstype.CertificatesV1beta1Interface
}

// NewCertsAPIRenawal takes a Kubernetes interface and returns a renewal Interface.
func NewCertsAPIRenawal(client kubernetes.Interface) Interface {
	return &CertsAPIRenewal{
		client: client.CertificatesV1beta1(),
	}
}

// Renew takes a certificate using the cert and key.
func (r *CertsAPIRenewal) Renew(cfg *certutil.Config) (*x509.Certificate, *rsa.PrivateKey, error) {
	reqTmp := &x509.CertificateRequest{
		Subject: pkix.Name{
			CommonName:   cfg.CommonName,
			Organization: cfg.Organization,
		},
		DNSNames:    cfg.AltNames.DNSNames,
		IPAddresses: cfg.AltNames.IPs,
	}

	key, err := certutil.NewPrivateKey()
	if err != nil {
		return nil, nil, errors.Wrap(err, "couldn't create new private key")
	}

	csr, err := x509.CreateCertificateRequest(rand.Reader, reqTmp, key)
	if err != nil {
		return nil, nil, errors.Wrap(err, "couldn't create certificate signing request")
	}

	usages := make([]certsapi.KeyUsage, len(cfg.Usages))
	for i, usage := range cfg.Usages {
		certsAPIUsage, ok := usageMap[usage]
		if !ok {
			return nil, nil, fmt.Errorf("unknown key usage: %v", usage)
		}
		usages[i] = certsAPIUsage
	}

	k8sCSR := &certsapi.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: certAPIPrefixName,
		},
		Spec: certsapi.CertificateSigningRequestSpec{
			Request: csr,
			Usages:  usages,
		},
	}

	req, err := r.client.CertificateSigningRequests().Create(k8sCSR)
	if err != nil {
		return nil, nil, errors.Wrap(err, "couldn't create certificate signing request")
	}

	watcher, err := r.client.CertificateSigningRequests().Watch(metav1.ListOptions{
		Watch:         true,
		FieldSelector: fields.Set{"metadata.name": req.Name}.String(),
	})
	if err != nil {
		return nil, nil, errors.Wrap(err, "couldn't watch for certificate creation")
	}
	defer watcher.Stop()

	select {
	case ev := <-watcher.ResultChan():
		if ev.Type != watch.Modified {
			return nil, nil, fmt.Errorf("unexpected event received: %q", ev.Type)
		}
	case <-time.After(watchTimeout):
		return nil, nil, errors.New("timeout trying to sign certificate")
	}

	req, err = r.client.CertificateSigningRequests().Get(req.Name, metav1.GetOptions{})
	if len(req.Status.Conditions) < 1 {
		return nil, nil, errors.New("certificate signing request has no statuses")
	}

	// TODO: under what circumstances are there more than one?
	if status := req.Status.Conditions[0].Type; status != certsapi.CertificateApproved {
		return nil, nil, fmt.Errorf("unexpected certificate status: %v", status)
	}

	cert, err := x509.ParseCertificate(req.Status.Certificate)
	if err != nil {
		return nil, nil, errors.Wrap(err, "couldn't parse issued certificate")
	}

	return cert, key, nil
}

var usageMap = map[x509.ExtKeyUsage]certsapi.KeyUsage{
	x509.ExtKeyUsageAny:                        certsapi.UsageAny,
	x509.ExtKeyUsageServerAuth:                 certsapi.UsageServerAuth,
	x509.ExtKeyUsageClientAuth:                 certsapi.UsageClientAuth,
	x509.ExtKeyUsageCodeSigning:                certsapi.UsageCodeSigning,
	x509.ExtKeyUsageEmailProtection:            certsapi.UsageEmailProtection,
	x509.ExtKeyUsageIPSECEndSystem:             certsapi.UsageIPsecEndSystem,
	x509.ExtKeyUsageIPSECTunnel:                certsapi.UsageIPsecTunnel,
	x509.ExtKeyUsageIPSECUser:                  certsapi.UsageIPsecUser,
	x509.ExtKeyUsageTimeStamping:               certsapi.UsageTimestamping,
	x509.ExtKeyUsageOCSPSigning:                certsapi.UsageOCSPSigning,
	x509.ExtKeyUsageMicrosoftServerGatedCrypto: certsapi.UsageMicrosoftSGC,
	x509.ExtKeyUsageNetscapeServerGatedCrypto:  certsapi.UsageNetscapSGC,
}
