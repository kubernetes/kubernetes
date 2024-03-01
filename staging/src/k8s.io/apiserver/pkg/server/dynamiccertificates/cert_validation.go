/*
Copyright 2023 The Kubernetes Authors.

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

package dynamiccertificates

import (
	"crypto/x509"
	"fmt"
	"time"
)

type CertificateValidator interface {
	Validate(c *x509.Certificate) []error
}

type certValidatorUnion []CertificateValidator

var _ CertificateValidator = &certValidatorUnion{}

func (u certValidatorUnion) Validate(c *x509.Certificate) []error {
	var errs []error
	for _, v := range u {
		errs = append(errs, v.Validate(c)...)
	}
	return errs
}

type ServerCertValidator struct{}

func (v *ServerCertValidator) Validate(c *x509.Certificate) []error {
	var errs []error

	if err := validateCertLifetime(c); err != nil {
		errs = append(errs, err)
	}

	if !hasSAN(c) {
		errs = append(errs, fmt.Errorf("missing the Subject Alternative Name extension"))
	}

	var serverAuthEKUFound bool
	for _, eku := range c.ExtKeyUsage {
		if eku == x509.ExtKeyUsageServerAuth {
			serverAuthEKUFound = true
			break
		}
	}
	if !serverAuthEKUFound {
		errs = append(errs, fmt.Errorf("missing ServerAuth extended key usage extension"))
	}

	return errs
}

type ClientCertValidator struct{}

func (v *ClientCertValidator) Validate(c *x509.Certificate) []error {
	var errs []error

	if err := validateCertLifetime(c); err != nil {
		errs = append(errs, err)
	}

	var clientAuthEKUFound bool
	for _, eku := range c.ExtKeyUsage {
		if eku == x509.ExtKeyUsageClientAuth {
			clientAuthEKUFound = true
			break
		}
	}
	if !clientAuthEKUFound {
		errs = append(errs, fmt.Errorf("missing ClientAuth extended key usage extension"))
	}

	return errs
}

func validateCertLifetime(cert *x509.Certificate) error {
	now := time.Now()
	if now.Before(cert.NotBefore) {
		return fmt.Errorf("not yet valid")
	}
	if now.After(cert.NotAfter) {
		return fmt.Errorf("expired")
	}
	return nil
}

func hasSAN(c *x509.Certificate) bool {
	sanOID := []int{2, 5, 29, 17}

	for _, e := range c.Extensions {
		if e.Id.Equal(sanOID) {
			return true
		}
	}
	return false
}
