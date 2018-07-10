/*
Copyright 2014 The Kubernetes Authors.

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
	"crypto/x509/pkix"
	"fmt"
	"net/http"
	"time"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
)

var clientCertificateExpirationHistogram = prometheus.NewHistogram(
	prometheus.HistogramOpts{
		Namespace: "apiserver",
		Subsystem: "client",
		Name:      "certificate_expiration_seconds",
		Help:      "Distribution of the remaining lifetime on the certificate used to authenticate a request.",
		Buckets: []float64{
			0,
			(6 * time.Hour).Seconds(),
			(12 * time.Hour).Seconds(),
			(24 * time.Hour).Seconds(),
			(2 * 24 * time.Hour).Seconds(),
			(4 * 24 * time.Hour).Seconds(),
			(7 * 24 * time.Hour).Seconds(),
			(30 * 24 * time.Hour).Seconds(),
			(3 * 30 * 24 * time.Hour).Seconds(),
			(6 * 30 * 24 * time.Hour).Seconds(),
			(12 * 30 * 24 * time.Hour).Seconds(),
		},
	},
)

func init() {
	prometheus.MustRegister(clientCertificateExpirationHistogram)
}

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
	if req.TLS == nil || len(req.TLS.PeerCertificates) == 0 {
		return nil, false, nil
	}

	// Use intermediates, if provided
	optsCopy := a.opts
	if optsCopy.Intermediates == nil && len(req.TLS.PeerCertificates) > 1 {
		optsCopy.Intermediates = x509.NewCertPool()
		for _, intermediate := range req.TLS.PeerCertificates[1:] {
			optsCopy.Intermediates.AddCert(intermediate)
		}
	}

	remaining := req.TLS.PeerCertificates[0].NotAfter.Sub(time.Now())
	clientCertificateExpirationHistogram.Observe(remaining.Seconds())
	chains, err := req.TLS.PeerCertificates[0].Verify(optsCopy)
	if err != nil {
		return nil, false, err
	}

	var errlist []error
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
	return nil, false, utilerrors.NewAggregate(errlist)
}

// Verifier implements request.Authenticator by verifying a client cert on the request, then delegating to the wrapped auth
type Verifier struct {
	opts x509.VerifyOptions
	auth authenticator.Request

	// allowedCommonNames contains the common names which a verified certificate is allowed to have.
	// If empty, all verified certificates are allowed.
	allowedCommonNames sets.String
}

// NewVerifier create a request.Authenticator by verifying a client cert on the request, then delegating to the wrapped auth
func NewVerifier(opts x509.VerifyOptions, auth authenticator.Request, allowedCommonNames sets.String) authenticator.Request {
	return &Verifier{opts, auth, allowedCommonNames}
}

// AuthenticateRequest verifies the presented client certificate, then delegates to the wrapped auth
func (a *Verifier) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	if req.TLS == nil || len(req.TLS.PeerCertificates) == 0 {
		return nil, false, nil
	}

	// Use intermediates, if provided
	optsCopy := a.opts
	if optsCopy.Intermediates == nil && len(req.TLS.PeerCertificates) > 1 {
		optsCopy.Intermediates = x509.NewCertPool()
		for _, intermediate := range req.TLS.PeerCertificates[1:] {
			optsCopy.Intermediates.AddCert(intermediate)
		}
	}

	if _, err := req.TLS.PeerCertificates[0].Verify(optsCopy); err != nil {
		return nil, false, err
	}
	if err := a.verifySubject(req.TLS.PeerCertificates[0].Subject); err != nil {
		return nil, false, err
	}
	return a.auth.AuthenticateRequest(req)
}

func (a *Verifier) verifySubject(subject pkix.Name) error {
	// No CN restrictions
	if len(a.allowedCommonNames) == 0 {
		return nil
	}
	// Enforce CN restrictions
	if a.allowedCommonNames.Has(subject.CommonName) {
		return nil
	}
	glog.Warningf("x509: subject with cn=%s is not in the allowed list: %v", subject.CommonName, a.allowedCommonNames.List())
	return fmt.Errorf("x509: subject with cn=%s is not allowed", subject.CommonName)
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
	return &user.DefaultInfo{
		Name:   chain[0].Subject.CommonName,
		Groups: chain[0].Subject.Organization,
	}, true, nil
})
