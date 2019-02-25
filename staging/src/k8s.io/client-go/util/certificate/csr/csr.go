/*
Copyright 2016 The Kubernetes Authors.

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

package csr

import (
	"context"
	"crypto"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"reflect"
	"time"

	"k8s.io/klog"

	certificates "k8s.io/api/certificates/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	certificatesclient "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	certutil "k8s.io/client-go/util/cert"
)

// RequestCertificate will either use an existing (if this process has run
// before but not to completion) or create a certificate signing request using the
// PEM encoded CSR and send it to API server, then it will watch the object's
// status, once approved by API server, it will return the API server's issued
// certificate (pem-encoded). If there is any errors, or the watch timeouts, it
// will return an error.
func RequestCertificate(client certificatesclient.CertificateSigningRequestInterface, csrData []byte, name string, usages []certificates.KeyUsage, privateKey interface{}) (req *certificates.CertificateSigningRequest, err error) {
	csr := &certificates.CertificateSigningRequest{
		// Username, UID, Groups will be injected by API server.
		TypeMeta: metav1.TypeMeta{Kind: "CertificateSigningRequest"},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: certificates.CertificateSigningRequestSpec{
			Request: csrData,
			Usages:  usages,
		},
	}
	if len(csr.Name) == 0 {
		csr.GenerateName = "csr-"
	}

	req, err = client.Create(csr)
	switch {
	case err == nil:
	case errors.IsAlreadyExists(err) && len(name) > 0:
		klog.Infof("csr for this node already exists, reusing")
		req, err = client.Get(name, metav1.GetOptions{})
		if err != nil {
			return nil, formatError("cannot retrieve certificate signing request: %v", err)
		}
		if err := ensureCompatible(req, csr, privateKey); err != nil {
			return nil, fmt.Errorf("retrieved csr is not compatible: %v", err)
		}
		klog.Infof("csr for this node is still valid")
	default:
		return nil, formatError("cannot create certificate signing request: %v", err)
	}
	return req, nil
}

// WaitForCertificate waits for a certificate to be issued until timeout, or returns an error.
func WaitForCertificate(client certificatesclient.CertificateSigningRequestInterface, req *certificates.CertificateSigningRequest, timeout time.Duration) (certData []byte, err error) {
	fieldSelector := fields.OneTermEqualSelector("metadata.name", req.Name).String()
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fieldSelector
			return client.List(options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fieldSelector
			return client.Watch(options)
		},
	}
	ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), timeout)
	defer cancel()
	event, err := watchtools.UntilWithSync(
		ctx,
		lw,
		&certificates.CertificateSigningRequest{},
		nil,
		func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Modified, watch.Added:
			case watch.Deleted:
				return false, fmt.Errorf("csr %q was deleted", req.Name)
			default:
				return false, nil
			}
			csr := event.Object.(*certificates.CertificateSigningRequest)
			if csr.UID != req.UID {
				return false, fmt.Errorf("csr %q changed UIDs", csr.Name)
			}
			for _, c := range csr.Status.Conditions {
				if c.Type == certificates.CertificateDenied {
					return false, fmt.Errorf("certificate signing request is not approved, reason: %v, message: %v", c.Reason, c.Message)
				}
				if c.Type == certificates.CertificateApproved && csr.Status.Certificate != nil {
					return true, nil
				}
			}
			return false, nil
		},
	)
	if err == wait.ErrWaitTimeout {
		return nil, wait.ErrWaitTimeout
	}
	if err != nil {
		return nil, formatError("cannot watch on the certificate signing request: %v", err)
	}

	return event.Object.(*certificates.CertificateSigningRequest).Status.Certificate, nil
}

// ensureCompatible ensures that a CSR object is compatible with an original CSR
func ensureCompatible(new, orig *certificates.CertificateSigningRequest, privateKey interface{}) error {
	newCSR, err := parseCSR(new)
	if err != nil {
		return fmt.Errorf("unable to parse new csr: %v", err)
	}
	origCSR, err := parseCSR(orig)
	if err != nil {
		return fmt.Errorf("unable to parse original csr: %v", err)
	}
	if !reflect.DeepEqual(newCSR.Subject, origCSR.Subject) {
		return fmt.Errorf("csr subjects differ: new: %#v, orig: %#v", newCSR.Subject, origCSR.Subject)
	}
	signer, ok := privateKey.(crypto.Signer)
	if !ok {
		return fmt.Errorf("privateKey is not a signer")
	}
	newCSR.PublicKey = signer.Public()
	if err := newCSR.CheckSignature(); err != nil {
		return fmt.Errorf("error validating signature new CSR against old key: %v", err)
	}
	if len(new.Status.Certificate) > 0 {
		certs, err := certutil.ParseCertsPEM(new.Status.Certificate)
		if err != nil {
			return fmt.Errorf("error parsing signed certificate for CSR: %v", err)
		}
		now := time.Now()
		for _, cert := range certs {
			if now.After(cert.NotAfter) {
				return fmt.Errorf("one of the certificates for the CSR has expired: %s", cert.NotAfter)
			}
		}
	}
	return nil
}

// formatError preserves the type of an API message but alters the message. Expects
// a single argument format string, and returns the wrapped error.
func formatError(format string, err error) error {
	if s, ok := err.(errors.APIStatus); ok {
		se := &errors.StatusError{ErrStatus: s.Status()}
		se.ErrStatus.Message = fmt.Sprintf(format, se.ErrStatus.Message)
		return se
	}
	return fmt.Errorf(format, err)
}

// parseCSR extracts the CSR from the API object and decodes it.
func parseCSR(obj *certificates.CertificateSigningRequest) (*x509.CertificateRequest, error) {
	// extract PEM from request object
	block, _ := pem.Decode(obj.Spec.Request)
	if block == nil || block.Type != "CERTIFICATE REQUEST" {
		return nil, fmt.Errorf("PEM block type must be CERTIFICATE REQUEST")
	}
	return x509.ParseCertificateRequest(block.Bytes)
}
