/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package x509

import (
	"crypto/x509"
	"net/http"

	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/util/errors"
)

// UserConversion defines an interface for extracting user info from a client certificate chain
type UserConversion interface {
	User(chain []*x509.Certificate) (user.Info, bool, error)
}

// UserConversionFunc is a function that implements the UserConversion interface.
type UserConversionFunc func(chain []*x509.Certificate) (user.Info, bool, error)

// User implements x509.UserConversion
func (f UserConversionFunc) User(chain []*x509.Certificate) (user.Info, bool, error) {
	return f(chain)
}

// Authenticator implements request.Authenticator by extracting user info from verified client certificates
type Authenticator struct {
	opts x509.VerifyOptions
	user UserConversion
}

// New returns a request.Authenticator that verifies client certificates using the provided
// VerifyOptions, and converts valid certificate chains into user.Info using the provided UserConversion
func New(opts x509.VerifyOptions, user UserConversion) *Authenticator {
	return &Authenticator{opts, user}
}

// AuthenticateRequest authenticates the request using presented client certificates
func (a *Authenticator) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	if req.TLS == nil {
		return nil, false, nil
	}

	var errlist []error
	for _, cert := range req.TLS.PeerCertificates {
		chains, err := cert.Verify(a.opts)
		if err != nil {
			errlist = append(errlist, err)
			continue
		}

		for _, chain := range chains {
			user, ok, err := a.user.User(chain)
			if err != nil {
				errlist = append(errlist, err)
				continue
			}

			if ok {
				return user, ok, err
			}
		}
	}
	return nil, false, errors.NewAggregate(errlist)
}

// DefaultVerifyOptions returns VerifyOptions that use the system root certificates, current time,
// and requires certificates to be valid for client auth (x509.ExtKeyUsageClientAuth)
func DefaultVerifyOptions() x509.VerifyOptions {
	return x509.VerifyOptions{
		KeyUsages: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
}

// CommonNameUserConversion builds user info from a certificate chain using the subject's CommonName
var CommonNameUserConversion = UserConversionFunc(func(chain []*x509.Certificate) (user.Info, bool, error) {
	if len(chain[0].Subject.CommonName) == 0 {
		return nil, false, nil
	}
	return &user.DefaultInfo{Name: chain[0].Subject.CommonName}, true, nil
})

// DNSNameUserConversion builds user info from a certificate chain using the first DNSName on the certificate
var DNSNameUserConversion = UserConversionFunc(func(chain []*x509.Certificate) (user.Info, bool, error) {
	if len(chain[0].DNSNames) == 0 {
		return nil, false, nil
	}
	return &user.DefaultInfo{Name: chain[0].DNSNames[0]}, true, nil
})

// EmailAddressUserConversion builds user info from a certificate chain using the first EmailAddress on the certificate
var EmailAddressUserConversion = UserConversionFunc(func(chain []*x509.Certificate) (user.Info, bool, error) {
	if len(chain[0].EmailAddresses) == 0 {
		return nil, false, nil
	}
	return &user.DefaultInfo{Name: chain[0].EmailAddresses[0]}, true, nil
})
