package sign

import (
	"bytes"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha1"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"strings"
	"time"
	"unicode"
)

// An AWSEpochTime wraps a time value providing JSON serialization needed for
// AWS Policy epoch time fields.
type AWSEpochTime struct {
	time.Time
}

// NewAWSEpochTime returns a new AWSEpochTime pointer wrapping the Go time provided.
func NewAWSEpochTime(t time.Time) *AWSEpochTime {
	return &AWSEpochTime{t}
}

// MarshalJSON serializes the epoch time as AWS Profile epoch time.
func (t AWSEpochTime) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`{"AWS:EpochTime":%d}`, t.UTC().Unix())), nil
}

// An IPAddress wraps an IPAddress source IP providing JSON serialization information
type IPAddress struct {
	SourceIP string `json:"AWS:SourceIp"`
}

// A Condition defines the restrictions for how a signed URL can be used.
type Condition struct {
	// Optional IP address mask the signed URL must be requested from.
	IPAddress *IPAddress `json:"IpAddress,omitempty"`

	// Optional date that the signed URL cannot be used until. It is invalid
	// to make requests with the signed URL prior to this date.
	DateGreaterThan *AWSEpochTime `json:",omitempty"`

	// Required date that the signed URL will expire. A DateLessThan is required
	// sign cloud front URLs
	DateLessThan *AWSEpochTime `json:",omitempty"`
}

// A Statement is a collection of conditions for resources
type Statement struct {
	// The Web or RTMP resource the URL will be signed for
	Resource string

	// The set of conditions for this resource
	Condition Condition
}

// A Policy defines the resources that a signed will be signed for.
//
// See the following page for more information on how policies are constructed.
// http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-creating-signed-url-custom-policy.html#private-content-custom-policy-statement
type Policy struct {
	// List of resource and condition statements.
	// Signed URLs should only provide a single statement.
	Statements []Statement `json:"Statement"`
}

// Override for testing to mock out usage of crypto/rand.Reader
var randReader = rand.Reader

// Sign will sign a policy using an RSA private key. It will return a base 64
// encoded signature and policy if no error is encountered.
//
// The signature and policy should be added to the signed URL following the
// guidelines in:
// http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-signed-urls.html
func (p *Policy) Sign(privKey *rsa.PrivateKey) (b64Signature, b64Policy []byte, err error) {
	if err = p.Validate(); err != nil {
		return nil, nil, err
	}

	// Build and escape the policy
	b64Policy, jsonPolicy, err := encodePolicy(p)
	if err != nil {
		return nil, nil, err
	}
	awsEscapeEncoded(b64Policy)

	// Build and escape the signature
	b64Signature, err = signEncodedPolicy(randReader, jsonPolicy, privKey)
	if err != nil {
		return nil, nil, err
	}
	awsEscapeEncoded(b64Signature)

	return b64Signature, b64Policy, nil
}

// Validate verifies that the policy is valid and usable, and returns an
// error if there is a problem.
func (p *Policy) Validate() error {
	if len(p.Statements) == 0 {
		return fmt.Errorf("at least one policy statement is required")
	}
	for i, s := range p.Statements {
		if s.Resource == "" {
			return fmt.Errorf("statement at index %d does not have a resource", i)
		}
		if !isASCII(s.Resource) {
			return fmt.Errorf("unable to sign resource, [%s]. "+
				"Resources must only contain ascii characters. "+
				"Hostnames with unicode should be encoded as Punycode, (e.g. golang.org/x/net/idna), "+
				"and URL unicode path/query characters should be escaped.", s.Resource)
		}
	}

	return nil
}

// CreateResource constructs, validates, and returns a resource URL string. An
// error will be returned if unable to create the resource string.
func CreateResource(scheme, u string) (string, error) {
	scheme = strings.ToLower(scheme)

	if scheme == "http" || scheme == "https" || scheme == "http*" || scheme == "*" {
		return u, nil
	}

	if scheme == "rtmp" {
		parsed, err := url.Parse(u)
		if err != nil {
			return "", fmt.Errorf("unable to parse rtmp URL, err: %s", err)
		}

		rtmpURL := strings.TrimLeft(parsed.Path, "/")
		if parsed.RawQuery != "" {
			rtmpURL = fmt.Sprintf("%s?%s", rtmpURL, parsed.RawQuery)
		}

		return rtmpURL, nil
	}

	return "", fmt.Errorf("invalid URL scheme must be http, https, or rtmp. Provided: %s", scheme)
}

// NewCannedPolicy returns a new Canned Policy constructed using the resource
// and expires time. This can be used to generate the basic model for a Policy
// that can be then augmented with additional conditions.
//
// See the following page for more information on how policies are constructed.
// http://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-creating-signed-url-custom-policy.html#private-content-custom-policy-statement
func NewCannedPolicy(resource string, expires time.Time) *Policy {
	return &Policy{
		Statements: []Statement{
			{
				Resource: resource,
				Condition: Condition{
					DateLessThan: NewAWSEpochTime(expires),
				},
			},
		},
	}
}

// encodePolicy encodes the Policy as JSON and also base 64 encodes it.
func encodePolicy(p *Policy) (b64Policy, jsonPolicy []byte, err error) {
	jsonPolicy, err = json.Marshal(p)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to encode policy, %s", err.Error())
	}

	// Remove leading and trailing white space, JSON encoding will note include
	// whitespace within the encoding.
	jsonPolicy = bytes.TrimSpace(jsonPolicy)

	b64Policy = make([]byte, base64.StdEncoding.EncodedLen(len(jsonPolicy)))
	base64.StdEncoding.Encode(b64Policy, jsonPolicy)
	return b64Policy, jsonPolicy, nil
}

// signEncodedPolicy will sign and base 64 encode the JSON encoded policy.
func signEncodedPolicy(randReader io.Reader, jsonPolicy []byte, privKey *rsa.PrivateKey) ([]byte, error) {
	hash := sha1.New()
	if _, err := bytes.NewReader(jsonPolicy).WriteTo(hash); err != nil {
		return nil, fmt.Errorf("failed to calculate signing hash, %s", err.Error())
	}

	sig, err := rsa.SignPKCS1v15(randReader, privKey, crypto.SHA1, hash.Sum(nil))
	if err != nil {
		return nil, fmt.Errorf("failed to sign policy, %s", err.Error())
	}

	b64Sig := make([]byte, base64.StdEncoding.EncodedLen(len(sig)))
	base64.StdEncoding.Encode(b64Sig, sig)
	return b64Sig, nil
}

// special characters to be replaced with awsEscapeEncoded
var invalidEncodedChar = map[byte]byte{
	'+': '-',
	'=': '_',
	'/': '~',
}

// awsEscapeEncoded will replace base64 encoding's special characters to be URL safe.
func awsEscapeEncoded(b []byte) {
	for i, v := range b {
		if r, ok := invalidEncodedChar[v]; ok {
			b[i] = r
		}
	}
}

func isASCII(u string) bool {
	for _, c := range u {
		if c > unicode.MaxASCII {
			return false
		}
	}
	return true
}
