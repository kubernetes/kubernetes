package sign

import (
	"crypto/rsa"
	"fmt"
	"net/http"
	"strings"
	"time"
)

const (
	// CookiePolicyName name of the policy cookie
	CookiePolicyName = "CloudFront-Policy"
	// CookieSignatureName name of the signature cookie
	CookieSignatureName = "CloudFront-Signature"
	// CookieKeyIDName name of the signing Key ID cookie
	CookieKeyIDName = "CloudFront-Key-Pair-Id"
)

// A CookieOptions optional additional options that can be applied to the signed
// cookies.
type CookieOptions struct {
	Path   string
	Domain string
	Secure bool
}

// apply will integration the options provided into the base cookie options
// a new copy will be returned. The base CookieOption will not be modified.
func (o CookieOptions) apply(opts ...func(*CookieOptions)) CookieOptions {
	if len(opts) == 0 {
		return o
	}

	for _, opt := range opts {
		opt(&o)
	}

	return o
}

// A CookieSigner provides signing utilities to sign Cookies for Amazon CloudFront
// resources. Using a private key and Credential Key Pair key ID the CookieSigner
// only needs to be created once per Credential Key Pair key ID and private key.
//
// More information about signed Cookies and their structure can be found at:
// http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-setting-signed-cookie-custom-policy.html
//
// To sign a Cookie, create a CookieSigner with your private key and credential
// pair key ID. Once you have a CookieSigner instance you can call Sign or
// SignWithPolicy to sign the URLs.
//
// The signer is safe to use concurrently, but the optional cookies options
// are not safe to modify concurrently.
type CookieSigner struct {
	keyID   string
	privKey *rsa.PrivateKey

	Opts CookieOptions
}

// NewCookieSigner constructs and returns a new CookieSigner to be used to for
// signing Amazon CloudFront URL resources with.
func NewCookieSigner(keyID string, privKey *rsa.PrivateKey, opts ...func(*CookieOptions)) *CookieSigner {
	signer := &CookieSigner{
		keyID:   keyID,
		privKey: privKey,
		Opts:    CookieOptions{}.apply(opts...),
	}

	return signer
}

// Sign returns the cookies needed to allow user agents to make arbetrary
// requests to cloudfront for the resource(s) defined by the policy.
//
// Sign will create a CloudFront policy with only a resource and condition of
// DateLessThan equal to the expires time provided.
//
// The returned slice cookies should all be added to the Client's cookies or
// server's response.
//
// Example:
//    s := sign.NewCookieSigner(keyID, privKey)
//
//    // Get Signed cookies for a resource that will expire in 1 hour
//    cookies, err := s.Sign("*", time.Now().Add(1 * time.Hour))
//    if err != nil {
//        fmt.Println("failed to create signed cookies", err)
//        return
//    }
//
//    // Or get Signed cookies for a resource that will expire in 1 hour
//    // and set path and domain of cookies
//    cookies, err := s.Sign("*", time.Now().Add(1 * time.Hour), func(o *sign.CookieOptions) {
//        o.Path = "/"
//        o.Domain = ".example.com"
//    })
//    if err != nil {
//        fmt.Println("failed to create signed cookies", err)
//        return
//    }
//
//    // Server Response via http.ResponseWriter
//    for _, c := range cookies {
//        http.SetCookie(w, c)
//    }
//
//    // Client request via the cookie jar
//    if client.CookieJar != nil {
//        for _, c := range cookies {
//           client.Cookie(w, c)
//        }
//    }
func (s CookieSigner) Sign(u string, expires time.Time, opts ...func(*CookieOptions)) ([]*http.Cookie, error) {
	scheme, err := cookieURLScheme(u)
	if err != nil {
		return nil, err
	}

	resource, err := CreateResource(scheme, u)
	if err != nil {
		return nil, err
	}

	p := NewCannedPolicy(resource, expires)
	return createCookies(p, s.keyID, s.privKey, s.Opts.apply(opts...))
}

// Returns and validates the URL's scheme.
// http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-setting-signed-cookie-custom-policy.html#private-content-custom-policy-statement-cookies
func cookieURLScheme(u string) (string, error) {
	parts := strings.SplitN(u, "://", 2)
	if len(parts) != 2 {
		return "", fmt.Errorf("invalid cookie URL, missing scheme")
	}

	scheme := strings.ToLower(parts[0])
	if scheme != "http" && scheme != "https" && scheme != "http*" {
		return "", fmt.Errorf("invalid cookie URL scheme. Expect http, https, or http*. Go, %s", scheme)
	}

	return scheme, nil
}

// SignWithPolicy returns the cookies needed to allow user agents to make
// arbetrairy requets to cloudfront for the resource(s) defined by the policy.
//
// The returned slice cookies should all be added to the Client's cookies or
// server's response.
//
// Example:
//    s := sign.NewCookieSigner(keyID, privKey)
//
//    policy := &sign.Policy{
//        Statements: []sign.Statement{
//            {
//                // Read the provided documentation on how to set this
//                // correctly, you'll probably want to use wildcards.
//                Resource: rawCloudFrontURL,
//                Condition: sign.Condition{
//                    // Optional IP source address range
//                    IPAddress: &sign.IPAddress{SourceIP: "192.0.2.0/24"},
//                    // Optional date URL is not valid until
//                    DateGreaterThan: &sign.AWSEpochTime{time.Now().Add(30 * time.Minute)},
//                    // Required date the URL will expire after
//                    DateLessThan: &sign.AWSEpochTime{time.Now().Add(1 * time.Hour)},
//                },
//            },
//        },
//    }
//
//    // Get Signed cookies for a resource that will expire in 1 hour
//    cookies, err := s.SignWithPolicy(policy)
//    if err != nil {
//        fmt.Println("failed to create signed cookies", err)
//        return
//    }
//
//    // Or get Signed cookies for a resource that will expire in 1 hour
//    // and set path and domain of cookies
//    cookies, err := s.Sign(policy, func(o *sign.CookieOptions) {
//        o.Path = "/"
//        o.Domain = ".example.com"
//    })
//    if err != nil {
//        fmt.Println("failed to create signed cookies", err)
//        return
//    }
//
//    // Server Response via http.ResponseWriter
//    for _, c := range cookies {
//        http.SetCookie(w, c)
//    }
//
//    // Client request via the cookie jar
//    if client.CookieJar != nil {
//        for _, c := range cookies {
//           client.Cookie(w, c)
//        }
//    }
func (s CookieSigner) SignWithPolicy(p *Policy, opts ...func(*CookieOptions)) ([]*http.Cookie, error) {
	return createCookies(p, s.keyID, s.privKey, s.Opts.apply(opts...))
}

// Prepares the cookies to be attached to the header. An (optional) options
// struct is provided in case people don't want to manually edit their cookies.
func createCookies(p *Policy, keyID string, privKey *rsa.PrivateKey, opt CookieOptions) ([]*http.Cookie, error) {
	b64Sig, b64Policy, err := p.Sign(privKey)
	if err != nil {
		return nil, err
	}

	// Creates proper cookies
	cPolicy := &http.Cookie{
		Name:     CookiePolicyName,
		Value:    string(b64Policy),
		HttpOnly: true,
	}
	cSignature := &http.Cookie{
		Name:     CookieSignatureName,
		Value:    string(b64Sig),
		HttpOnly: true,
	}
	cKey := &http.Cookie{
		Name:     CookieKeyIDName,
		Value:    keyID,
		HttpOnly: true,
	}

	cookies := []*http.Cookie{cPolicy, cSignature, cKey}

	// Applie the cookie options
	for _, c := range cookies {
		c.Path = opt.Path
		c.Domain = opt.Domain
		c.Secure = opt.Secure
	}

	return cookies, nil
}
