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
	"context"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/hex"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

/*
 * By default, the following metric is defined as falling under
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/1209-metrics-stability/20190404-kubernetes-control-plane-metrics-stability.md#stability-classes)
 *
 * Promoting the stability level of the metric is a responsibility of the component owner, since it
 * involves explicitly acknowledging support for the metric across multiple releases, in accordance with
 * the metric stability policy.
 */
var clientCertificateExpirationHistogram = metrics.NewHistogram(
	&metrics.HistogramOpts{
		Namespace: "apiserver",
		Subsystem: "client",
		Name:      "certificate_expiration_seconds",
		Help:      "Distribution of the remaining lifetime on the certificate used to authenticate a request.",
		Buckets: []float64{
			0,
			(30 * time.Minute).Seconds(),
			(1 * time.Hour).Seconds(),
			(2 * time.Hour).Seconds(),
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
		StabilityLevel: metrics.ALPHA,
	},
)

func init() {
	legacyregistry.MustRegister(clientCertificateExpirationHistogram)
}

// UserConversion defines an interface for extracting user info from a client certificate chain
type UserConversion interface {
	User(chain []*x509.Certificate) (*authenticator.Response, bool, error)
}

// UserConversionFunc is a function that implements the UserConversion interface.
type UserConversionFunc func(chain []*x509.Certificate) (*authenticator.Response, bool, error)

// User implements x509.UserConversion
func (f UserConversionFunc) User(chain []*x509.Certificate) (*authenticator.Response, bool, error) {
	return f(chain)
}

func columnSeparatedHex(d []byte) string {
	h := strings.ToUpper(hex.EncodeToString(d))
	var sb strings.Builder
	for i, r := range h {
		sb.WriteRune(r)
		if i%2 == 1 && i != len(h)-1 {
			sb.WriteRune(':')
		}
	}
	return sb.String()
}

func certificateIdentifier(c *x509.Certificate) string {
	return fmt.Sprintf(
		"SN=%d, SKID=%s, AKID=%s",
		c.SerialNumber,
		columnSeparatedHex(c.SubjectKeyId),
		columnSeparatedHex(c.AuthorityKeyId),
	)
}

// VerifyOptionFunc is function which provides a shallow copy of the VerifyOptions to the authenticator.  This allows
// for cases where the options (particularly the CAs) can change.  If the bool is false, then the returned VerifyOptions
// are ignored and the authenticator will express "no opinion".  This allows a clear signal for cases where a CertPool
// is eventually expected, but not currently present.
type VerifyOptionFunc func() (x509.VerifyOptions, bool)

// Authenticator implements request.Authenticator by extracting user info from verified client certificates
type Authenticator struct {
	*certificateVerifier
	user UserConversion
}

// New returns a request.Authenticator that verifies client certificates using the provided
// VerifyOptions, and converts valid certificate chains into user.Info using the provided UserConversion
func New(opts x509.VerifyOptions, user UserConversion) *Authenticator {
	return NewDynamic(StaticVerifierFn(opts), user)
}

// NewDynamic returns a request.Authenticator that verifies client certificates using the provided
// VerifyOptionFunc (which may be dynamic), and converts valid certificate chains into user.Info using the provided UserConversion
func NewDynamic(caProvider dynamiccertificates.CAContentProvider, user UserConversion) *Authenticator {
	return &Authenticator{certificateVerifier: newCertificateVerifier(caProvider), user: user}
}

// AuthenticateRequest authenticates the request using presented client certificates
func (a *Authenticator) AuthenticateRequest(req *http.Request) (*authenticator.Response, bool, error) {
	if req.TLS == nil || len(req.TLS.PeerCertificates) == 0 {
		return nil, false, nil
	}

	optsCopy, ok := a.caProvider.VerifyOptions()
	if !ok {
		// if there are intentionally no verify options, then we cannot
		// authenticate this request.
		return nil, false, nil
	}

	remaining := req.TLS.PeerCertificates[0].NotAfter.Sub(time.Now())
	clientCertificateExpirationHistogram.WithContext(req.Context()).Observe(remaining.Seconds())

	chains, err := a.verifyClientCerts(req.Context(), optsCopy, req.TLS.PeerCertificates)
	if err != nil {
		return nil, false, fmt.Errorf("verifying certificate %s failed: %w", certificateIdentifier(req.TLS.PeerCertificates[0]), err)
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
	*certificateVerifier
	auth authenticator.Request

	// allowedCommonNames contains the common names which a verified certificate is allowed to have.
	// If empty, all verified certificates are allowed.
	allowedCommonNames StringSliceProvider
}

// NewVerifier create a request.Authenticator by verifying a client cert on the request, then delegating to the wrapped auth
func NewVerifier(opts x509.VerifyOptions, auth authenticator.Request, allowedCommonNames sets.String) *Verifier {
	return NewDynamicCAVerifier(StaticVerifierFn(opts), auth, StaticStringSlice(allowedCommonNames.List()))
}

// NewDynamicCAVerifier create a request.Authenticator by verifying a client cert on the request, then delegating to the wrapped auth
func NewDynamicCAVerifier(caProvider dynamiccertificates.CAContentProvider, auth authenticator.Request, allowedCommonNames StringSliceProvider) *Verifier {
	return &Verifier{certificateVerifier: newCertificateVerifier(caProvider), auth: auth, allowedCommonNames: allowedCommonNames}
}

// AuthenticateRequest verifies the presented client certificate, then delegates to the wrapped auth
func (a *Verifier) AuthenticateRequest(req *http.Request) (*authenticator.Response, bool, error) {
	if req.TLS == nil || len(req.TLS.PeerCertificates) == 0 {
		return nil, false, nil
	}

	optsCopy, ok := a.caProvider.VerifyOptions()
	if !ok {
		// if there are intentionally no verify options, then we cannot
		// authenticate this request.
		return nil, false, nil
	}

	if _, err := a.verifyClientCerts(req.Context(), optsCopy, req.TLS.PeerCertificates); err != nil {
		return nil, false, fmt.Errorf("verifying certificate %s failed: %w", certificateIdentifier(req.TLS.PeerCertificates[0]), err)
	}

	if err := a.verifySubject(req.TLS.PeerCertificates[0].Subject); err != nil {
		return nil, false, err
	}
	return a.auth.AuthenticateRequest(req)
}

func (a *Verifier) verifySubject(subject pkix.Name) error {
	// No CN restrictions
	if len(a.allowedCommonNames.Value()) == 0 {
		return nil
	}
	// Enforce CN restrictions
	for _, allowedCommonName := range a.allowedCommonNames.Value() {
		if allowedCommonName == subject.CommonName {
			return nil
		}
	}
	return fmt.Errorf("x509: subject with cn=%s is not in the allowed list", subject.CommonName)
}

// CommonNameUserConversion builds user info from a certificate chain using the subject's CommonName
var CommonNameUserConversion = UserConversionFunc(func(chain []*x509.Certificate) (*authenticator.Response, bool, error) {
	if len(chain[0].Subject.CommonName) == 0 {
		return nil, false, nil
	}
	return &authenticator.Response{
		User: &user.DefaultInfo{
			Name:   chain[0].Subject.CommonName,
			Groups: chain[0].Subject.Organization,
		},
	}, true, nil
})

func newCertificateVerifier(caProvider dynamiccertificates.CAContentProvider) *certificateVerifier {
	out := &certificateVerifier{
		caProvider: caProvider,
		// We start at 1 since it's not the zero value
		// cachedVerification.generation.
		generation: 1,
	}
	caProvider.AddListener(out)
	out.ctxCacheKey = cachedVerificationCtxKey(unsafe.Pointer(out))
	return out
}

type certificateVerifier struct {
	caProvider  dynamiccertificates.CAContentProvider
	generation  uint64
	ctxCacheKey cachedVerificationCtxKey
}

func (cv *certificateVerifier) verifyClientCerts(ctx context.Context, opts x509.VerifyOptions, peerCertificates []*x509.Certificate) ([][]*x509.Certificate, error) {
	verify := func() ([][]*x509.Certificate, error) {
		// Use intermediates, if provided
		if opts.Intermediates == nil && len(peerCertificates) > 1 {
			opts.Intermediates = x509.NewCertPool()
			for _, intermediate := range peerCertificates[1:] {
				opts.Intermediates.AddCert(intermediate)
			}
		}

		return peerCertificates[0].Verify(opts)
	}

	entry, ok := ctx.Value(cv.ctxCacheKey).(*cachedVerification)
	if !ok {
		return verify()
	}

	if opts.CurrentTime.IsZero() {
		opts.CurrentTime = time.Now()
	}
	currentGeneration := atomic.LoadUint64(&cv.generation)

	entry.Lock()
	defer entry.Unlock()

	if !entry.stale(currentGeneration, opts.CurrentTime) {
		return entry.chains, entry.err
	}

	entry.chains, entry.err = verify()
	entry.generation = currentGeneration
	entry.cacheUntil = calculateCacheUntil(opts.CurrentTime, entry.chains, peerCertificates)

	return entry.chains, entry.err
}

func (cv *certificateVerifier) WithTLSVerificationCache(ctx context.Context, _ net.Conn) context.Context {
	return context.WithValue(ctx, cv.ctxCacheKey, &cachedVerification{})
}

func (cv *certificateVerifier) Enqueue() {
	atomic.AddUint64(&cv.generation, 1)
}

func calculateCacheUntil(now time.Time, chains [][]*x509.Certificate, peerCertificates []*x509.Certificate) time.Time {
	if chains != nil {
		// We found valid chains. Calculate the latest point that a valid chain
		// will expire.
		var validUntil time.Time

		for _, chain := range chains {
			// Initialize notAfter for the chain using the leaf, with the added
			// benefit of panicking if we get handed a chain of length 0.
			chainNotAfter := chain[0].NotAfter

			// Find the earliest expiration.
			for _, cert := range chain[1:] {
				if chainNotAfter.After(cert.NotAfter) {
					chainNotAfter = cert.NotAfter
				}
			}

			// Swap if it's later then the current validUntil
			if chainNotAfter.After(validUntil) {
				validUntil = chainNotAfter
			}
		}

		return validUntil
	}

	// We don't have a valid chain, so we are caching an error and all we have
	// are the peer certs.

	// Find the latest not before.
	notBefore := peerCertificates[0].NotBefore
	for _, cert := range peerCertificates[1:] {
		if cert.NotBefore.After(notBefore) {
			notBefore = cert.NotBefore
		}
	}
	if notBefore.After(now) {
		// There's one or more certs in here that are not valid yet. Let's reverify
		// once all of the NotBefores have passed.
		return notBefore
	}

	// All certs are passed their NotBefore and verification still failed. Maybe
	// they are expired or maybe we don't have anchors for them. Either way,
	// these aren't going to start working until our CA changes. Cache error for
	// arbitrary amount of time. This could probably be "infinite future", but an
	// hour seems reasonable.
	return now.Add(1 * time.Hour)
}

type cachedVerificationCtxKey uintptr

type cachedVerification struct {
	sync.Mutex
	generation uint64
	cacheUntil time.Time

	chains [][]*x509.Certificate
	err    error
}

func (vc *cachedVerification) stale(generation uint64, now time.Time) bool {
	if vc.generation != generation {
		return true
	}
	if now.After(vc.cacheUntil) {
		return true
	}
	return false
}
