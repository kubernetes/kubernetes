package token

import (
	"crypto"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/auth"
	"github.com/docker/libtrust"
)

// accessSet maps a typed, named resource to
// a set of actions requested or authorized.
type accessSet map[auth.Resource]actionSet

// newAccessSet constructs an accessSet from
// a variable number of auth.Access items.
func newAccessSet(accessItems ...auth.Access) accessSet {
	accessSet := make(accessSet, len(accessItems))

	for _, access := range accessItems {
		resource := auth.Resource{
			Type: access.Type,
			Name: access.Name,
		}

		set, exists := accessSet[resource]
		if !exists {
			set = newActionSet()
			accessSet[resource] = set
		}

		set.add(access.Action)
	}

	return accessSet
}

// contains returns whether or not the given access is in this accessSet.
func (s accessSet) contains(access auth.Access) bool {
	actionSet, ok := s[access.Resource]
	if ok {
		return actionSet.contains(access.Action)
	}

	return false
}

// scopeParam returns a collection of scopes which can
// be used for a WWW-Authenticate challenge parameter.
// See https://tools.ietf.org/html/rfc6750#section-3
func (s accessSet) scopeParam() string {
	scopes := make([]string, 0, len(s))

	for resource, actionSet := range s {
		actions := strings.Join(actionSet.keys(), ",")
		scopes = append(scopes, fmt.Sprintf("%s:%s:%s", resource.Type, resource.Name, actions))
	}

	return strings.Join(scopes, " ")
}

// Errors used and exported by this package.
var (
	ErrInsufficientScope = errors.New("insufficient scope")
	ErrTokenRequired     = errors.New("authorization token required")
)

// authChallenge implements the auth.Challenge interface.
type authChallenge struct {
	err       error
	realm     string
	service   string
	accessSet accessSet
}

var _ auth.Challenge = authChallenge{}

// Error returns the internal error string for this authChallenge.
func (ac authChallenge) Error() string {
	return ac.err.Error()
}

// Status returns the HTTP Response Status Code for this authChallenge.
func (ac authChallenge) Status() int {
	return http.StatusUnauthorized
}

// challengeParams constructs the value to be used in
// the WWW-Authenticate response challenge header.
// See https://tools.ietf.org/html/rfc6750#section-3
func (ac authChallenge) challengeParams() string {
	str := fmt.Sprintf("Bearer realm=%q,service=%q", ac.realm, ac.service)

	if scope := ac.accessSet.scopeParam(); scope != "" {
		str = fmt.Sprintf("%s,scope=%q", str, scope)
	}

	if ac.err == ErrInvalidToken || ac.err == ErrMalformedToken {
		str = fmt.Sprintf("%s,error=%q", str, "invalid_token")
	} else if ac.err == ErrInsufficientScope {
		str = fmt.Sprintf("%s,error=%q", str, "insufficient_scope")
	}

	return str
}

// SetChallenge sets the WWW-Authenticate value for the response.
func (ac authChallenge) SetHeaders(w http.ResponseWriter) {
	w.Header().Add("WWW-Authenticate", ac.challengeParams())
}

// accessController implements the auth.AccessController interface.
type accessController struct {
	realm       string
	issuer      string
	service     string
	rootCerts   *x509.CertPool
	trustedKeys map[string]libtrust.PublicKey
}

// tokenAccessOptions is a convenience type for handling
// options to the contstructor of an accessController.
type tokenAccessOptions struct {
	realm          string
	issuer         string
	service        string
	rootCertBundle string
}

// checkOptions gathers the necessary options
// for an accessController from the given map.
func checkOptions(options map[string]interface{}) (tokenAccessOptions, error) {
	var opts tokenAccessOptions

	keys := []string{"realm", "issuer", "service", "rootcertbundle"}
	vals := make([]string, 0, len(keys))
	for _, key := range keys {
		val, ok := options[key].(string)
		if !ok {
			return opts, fmt.Errorf("token auth requires a valid option string: %q", key)
		}
		vals = append(vals, val)
	}

	opts.realm, opts.issuer, opts.service, opts.rootCertBundle = vals[0], vals[1], vals[2], vals[3]

	return opts, nil
}

// newAccessController creates an accessController using the given options.
func newAccessController(options map[string]interface{}) (auth.AccessController, error) {
	config, err := checkOptions(options)
	if err != nil {
		return nil, err
	}

	fp, err := os.Open(config.rootCertBundle)
	if err != nil {
		return nil, fmt.Errorf("unable to open token auth root certificate bundle file %q: %s", config.rootCertBundle, err)
	}
	defer fp.Close()

	rawCertBundle, err := ioutil.ReadAll(fp)
	if err != nil {
		return nil, fmt.Errorf("unable to read token auth root certificate bundle file %q: %s", config.rootCertBundle, err)
	}

	var rootCerts []*x509.Certificate
	pemBlock, rawCertBundle := pem.Decode(rawCertBundle)
	for pemBlock != nil {
		if pemBlock.Type == "CERTIFICATE" {
			cert, err := x509.ParseCertificate(pemBlock.Bytes)
			if err != nil {
				return nil, fmt.Errorf("unable to parse token auth root certificate: %s", err)
			}

			rootCerts = append(rootCerts, cert)
		}

		pemBlock, rawCertBundle = pem.Decode(rawCertBundle)
	}

	if len(rootCerts) == 0 {
		return nil, errors.New("token auth requires at least one token signing root certificate")
	}

	rootPool := x509.NewCertPool()
	trustedKeys := make(map[string]libtrust.PublicKey, len(rootCerts))
	for _, rootCert := range rootCerts {
		rootPool.AddCert(rootCert)
		pubKey, err := libtrust.FromCryptoPublicKey(crypto.PublicKey(rootCert.PublicKey))
		if err != nil {
			return nil, fmt.Errorf("unable to get public key from token auth root certificate: %s", err)
		}
		trustedKeys[pubKey.KeyID()] = pubKey
	}

	return &accessController{
		realm:       config.realm,
		issuer:      config.issuer,
		service:     config.service,
		rootCerts:   rootPool,
		trustedKeys: trustedKeys,
	}, nil
}

// Authorized handles checking whether the given request is authorized
// for actions on resources described by the given access items.
func (ac *accessController) Authorized(ctx context.Context, accessItems ...auth.Access) (context.Context, error) {
	challenge := &authChallenge{
		realm:     ac.realm,
		service:   ac.service,
		accessSet: newAccessSet(accessItems...),
	}

	req, err := context.GetRequest(ctx)
	if err != nil {
		return nil, err
	}

	parts := strings.Split(req.Header.Get("Authorization"), " ")

	if len(parts) != 2 || strings.ToLower(parts[0]) != "bearer" {
		challenge.err = ErrTokenRequired
		return nil, challenge
	}

	rawToken := parts[1]

	token, err := NewToken(rawToken)
	if err != nil {
		challenge.err = err
		return nil, challenge
	}

	verifyOpts := VerifyOptions{
		TrustedIssuers:    []string{ac.issuer},
		AcceptedAudiences: []string{ac.service},
		Roots:             ac.rootCerts,
		TrustedKeys:       ac.trustedKeys,
	}

	if err = token.Verify(verifyOpts); err != nil {
		challenge.err = err
		return nil, challenge
	}

	accessSet := token.accessSet()
	for _, access := range accessItems {
		if !accessSet.contains(access) {
			challenge.err = ErrInsufficientScope
			return nil, challenge
		}
	}

	ctx = auth.WithResources(ctx, token.resources())

	return auth.WithUser(ctx, auth.UserInfo{Name: token.Claims.Subject}), nil
}

// init handles registering the token auth backend.
func init() {
	auth.Register("token", auth.InitFunc(newAccessController))
}
