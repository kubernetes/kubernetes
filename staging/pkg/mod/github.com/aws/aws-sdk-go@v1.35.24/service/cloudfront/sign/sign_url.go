// Package sign provides utilities to generate signed URLs for Amazon CloudFront.
//
// More information about signed URLs and their structure can be found at:
// http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-creating-signed-url-canned-policy.html
//
// To sign a URL create a URLSigner with your private key and credential pair key ID.
// Once you have a URLSigner instance you can call Sign or SignWithPolicy to
// sign the URLs.
//
// Example:
//
//    // Sign URL to be valid for 1 hour from now.
//    signer := sign.NewURLSigner(keyID, privKey)
//    signedURL, err := signer.Sign(rawURL, time.Now().Add(1*time.Hour))
//    if err != nil {
//        log.Fatalf("Failed to sign url, err: %s\n", err.Error())
//    }
//
package sign

import (
	"crypto/rsa"
	"fmt"
	"net/url"
	"strings"
	"time"
)

// An URLSigner provides URL signing utilities to sign URLs for Amazon CloudFront
// resources. Using a private key and Credential Key Pair key ID the URLSigner
// only needs to be created once per Credential Key Pair key ID and private key.
//
// The signer is safe to use concurrently.
type URLSigner struct {
	keyID   string
	privKey *rsa.PrivateKey
}

// NewURLSigner constructs and returns a new URLSigner to be used to for signing
// Amazon CloudFront URL resources with.
func NewURLSigner(keyID string, privKey *rsa.PrivateKey) *URLSigner {
	return &URLSigner{
		keyID:   keyID,
		privKey: privKey,
	}
}

// Sign will sign a single URL to expire at the time of expires sign using the
// Amazon CloudFront default Canned Policy. The URL will be signed with the
// private key and Credential Key Pair Key ID previously provided to URLSigner.
//
// This is the default method of signing Amazon CloudFront URLs. If extra policy
// conditions are need other than URL expiry use SignWithPolicy instead.
//
// Example:
//
//    // Sign URL to be valid for 1 hour from now.
//    signer := sign.NewURLSigner(keyID, privKey)
//    signedURL, err := signer.Sign(rawURL, time.Now().Add(1*time.Hour))
//    if err != nil {
//        log.Fatalf("Failed to sign url, err: %s\n", err.Error())
//    }
//
func (s URLSigner) Sign(url string, expires time.Time) (string, error) {
	scheme, cleanedURL, err := cleanURLScheme(url)
	if err != nil {
		return "", err
	}

	resource, err := CreateResource(scheme, url)
	if err != nil {
		return "", err
	}

	return signURL(scheme, cleanedURL, s.keyID, NewCannedPolicy(resource, expires), false, s.privKey)
}

// SignWithPolicy will sign a URL with the Policy provided.  The URL will be
// signed with the private key and Credential Key Pair Key ID previously provided to URLSigner.
//
// Use this signing method if you are looking to sign a URL with more than just
// the URL's expiry time, or reusing Policies between multiple URL signings.
// If only the expiry time is needed you can use Sign and provide just the
// URL's expiry time. A minimum of at least one policy statement is required for a signed URL.
//
// Note: It is not safe to use Polices between multiple signers concurrently
//
// Example:
//
//     // Sign URL to be valid for 30 minutes from now, expires one hour from now, and
//     // restricted to the 192.0.2.0/24 IP address range.
//     policy := &sign.Policy{
//         Statements: []sign.Statement{
//             {
//                 Resource: rawURL,
//                 Condition: sign.Condition{
//                     // Optional IP source address range
//                     IPAddress: &sign.IPAddress{SourceIP: "192.0.2.0/24"},
//                     // Optional date URL is not valid until
//                     DateGreaterThan: &sign.AWSEpochTime{time.Now().Add(30 * time.Minute)},
//                     // Required date the URL will expire after
//                     DateLessThan: &sign.AWSEpochTime{time.Now().Add(1 * time.Hour)},
//                 },
//             },
//         },
//     }
//
//     signer := sign.NewURLSigner(keyID, privKey)
//     signedURL, err := signer.SignWithPolicy(rawURL, policy)
//     if err != nil {
//         log.Fatalf("Failed to sign url, err: %s\n", err.Error())
//     }
//
func (s URLSigner) SignWithPolicy(url string, p *Policy) (string, error) {
	scheme, cleanedURL, err := cleanURLScheme(url)
	if err != nil {
		return "", err
	}

	return signURL(scheme, cleanedURL, s.keyID, p, true, s.privKey)
}

func signURL(scheme, url, keyID string, p *Policy, customPolicy bool, privKey *rsa.PrivateKey) (string, error) {
	// Validation URL elements
	if err := validateURL(url); err != nil {
		return "", err
	}

	b64Signature, b64Policy, err := p.Sign(privKey)
	if err != nil {
		return "", err
	}

	// build and return signed URL
	builtURL := buildSignedURL(url, keyID, p, customPolicy, b64Policy, b64Signature)
	if scheme == "rtmp" {
		return buildRTMPURL(builtURL)
	}

	return builtURL, nil
}

func buildSignedURL(baseURL, keyID string, p *Policy, customPolicy bool, b64Policy, b64Signature []byte) string {
	pred := "?"
	if strings.Contains(baseURL, "?") {
		pred = "&"
	}
	signedURL := baseURL + pred

	if customPolicy {
		signedURL += "Policy=" + string(b64Policy)
	} else {
		signedURL += fmt.Sprintf("Expires=%d", p.Statements[0].Condition.DateLessThan.UTC().Unix())
	}
	signedURL += fmt.Sprintf("&Signature=%s&Key-Pair-Id=%s", string(b64Signature), keyID)

	return signedURL
}

func buildRTMPURL(u string) (string, error) {
	parsed, err := url.Parse(u)
	if err != nil {
		return "", fmt.Errorf("unable to parse rtmp signed URL, err: %s", err)
	}

	rtmpURL := strings.TrimLeft(parsed.Path, "/")
	if parsed.RawQuery != "" {
		rtmpURL = fmt.Sprintf("%s?%s", rtmpURL, parsed.RawQuery)
	}

	return rtmpURL, nil
}

func cleanURLScheme(u string) (scheme, cleanedURL string, err error) {
	parts := strings.SplitN(u, "://", 2)
	if len(parts) != 2 {
		return "", "", fmt.Errorf("invalid URL, missing scheme and domain/path")
	}
	scheme = strings.Replace(parts[0], "*", "", 1)
	cleanedURL = fmt.Sprintf("%s://%s", scheme, parts[1])

	return strings.ToLower(scheme), cleanedURL, nil
}

var illegalQueryParms = []string{"Expires", "Policy", "Signature", "Key-Pair-Id"}

func validateURL(u string) error {
	parsed, err := url.Parse(u)
	if err != nil {
		return fmt.Errorf("unable to parse URL, err: %s", err.Error())
	}

	if parsed.Scheme == "" {
		return fmt.Errorf("URL missing valid scheme, %s", u)
	}

	q := parsed.Query()
	for _, p := range illegalQueryParms {
		if _, ok := q[p]; ok {
			return fmt.Errorf("%s cannot be a query parameter for a signed URL", p)
		}
	}

	return nil
}
