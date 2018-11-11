// Package v4 implements signing for AWS V4 signer
//
// Provides request signing for request that need to be signed with
// AWS V4 Signatures.
//
// Standalone Signer
//
// Generally using the signer outside of the SDK should not require any additional
// logic when using Go v1.5 or higher. The signer does this by taking advantage
// of the URL.EscapedPath method. If your request URI requires additional escaping
// you many need to use the URL.Opaque to define what the raw URI should be sent
// to the service as.
//
// The signer will first check the URL.Opaque field, and use its value if set.
// The signer does require the URL.Opaque field to be set in the form of:
//
//     "//<hostname>/<path>"
//
//     // e.g.
//     "//example.com/some/path"
//
// The leading "//" and hostname are required or the URL.Opaque escaping will
// not work correctly.
//
// If URL.Opaque is not set the signer will fallback to the URL.EscapedPath()
// method and using the returned value. If you're using Go v1.4 you must set
// URL.Opaque if the URI path needs escaping. If URL.Opaque is not set with
// Go v1.5 the signer will fallback to URL.Path.
//
// AWS v4 signature validation requires that the canonical string's URI path
// element must be the URI escaped form of the HTTP request's path.
// http://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html
//
// The Go HTTP client will perform escaping automatically on the request. Some
// of these escaping may cause signature validation errors because the HTTP
// request differs from the URI path or query that the signature was generated.
// https://golang.org/pkg/net/url/#URL.EscapedPath
//
// Because of this, it is recommended that when using the signer outside of the
// SDK that explicitly escaping the request prior to being signed is preferable,
// and will help prevent signature validation errors. This can be done by setting
// the URL.Opaque or URL.RawPath. The SDK will use URL.Opaque first and then
// call URL.EscapedPath() if Opaque is not set.
//
// If signing a request intended for HTTP2 server, and you're using Go 1.6.2
// through 1.7.4 you should use the URL.RawPath as the pre-escaped form of the
// request URL. https://github.com/golang/go/issues/16847 points to a bug in
// Go pre 1.8 that fails to make HTTP2 requests using absolute URL in the HTTP
// message. URL.Opaque generally will force Go to make requests with absolute URL.
// URL.RawPath does not do this, but RawPath must be a valid escaping of Path
// or url.EscapedPath will ignore the RawPath escaping.
//
// Test `TestStandaloneSign` provides a complete example of using the signer
// outside of the SDK and pre-escaping the URI path.
package v4

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/sdkio"
	"github.com/aws/aws-sdk-go/private/protocol/rest"
)

const (
	authHeaderPrefix = "AWS4-HMAC-SHA256"
	timeFormat       = "20060102T150405Z"
	shortTimeFormat  = "20060102"

	// emptyStringSHA256 is a SHA256 of an empty string
	emptyStringSHA256 = `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
)

var ignoredHeaders = rules{
	blacklist{
		mapRule{
			"Authorization":   struct{}{},
			"User-Agent":      struct{}{},
			"X-Amzn-Trace-Id": struct{}{},
		},
	},
}

// requiredSignedHeaders is a whitelist for build canonical headers.
var requiredSignedHeaders = rules{
	whitelist{
		mapRule{
			"Cache-Control":                                               struct{}{},
			"Content-Disposition":                                         struct{}{},
			"Content-Encoding":                                            struct{}{},
			"Content-Language":                                            struct{}{},
			"Content-Md5":                                                 struct{}{},
			"Content-Type":                                                struct{}{},
			"Expires":                                                     struct{}{},
			"If-Match":                                                    struct{}{},
			"If-Modified-Since":                                           struct{}{},
			"If-None-Match":                                               struct{}{},
			"If-Unmodified-Since":                                         struct{}{},
			"Range":                                                       struct{}{},
			"X-Amz-Acl":                                                   struct{}{},
			"X-Amz-Copy-Source":                                           struct{}{},
			"X-Amz-Copy-Source-If-Match":                                  struct{}{},
			"X-Amz-Copy-Source-If-Modified-Since":                         struct{}{},
			"X-Amz-Copy-Source-If-None-Match":                             struct{}{},
			"X-Amz-Copy-Source-If-Unmodified-Since":                       struct{}{},
			"X-Amz-Copy-Source-Range":                                     struct{}{},
			"X-Amz-Copy-Source-Server-Side-Encryption-Customer-Algorithm": struct{}{},
			"X-Amz-Copy-Source-Server-Side-Encryption-Customer-Key":       struct{}{},
			"X-Amz-Copy-Source-Server-Side-Encryption-Customer-Key-Md5":   struct{}{},
			"X-Amz-Grant-Full-control":                                    struct{}{},
			"X-Amz-Grant-Read":                                            struct{}{},
			"X-Amz-Grant-Read-Acp":                                        struct{}{},
			"X-Amz-Grant-Write":                                           struct{}{},
			"X-Amz-Grant-Write-Acp":                                       struct{}{},
			"X-Amz-Metadata-Directive":                                    struct{}{},
			"X-Amz-Mfa":                                                   struct{}{},
			"X-Amz-Request-Payer":                                         struct{}{},
			"X-Amz-Server-Side-Encryption":                                struct{}{},
			"X-Amz-Server-Side-Encryption-Aws-Kms-Key-Id":                 struct{}{},
			"X-Amz-Server-Side-Encryption-Customer-Algorithm":             struct{}{},
			"X-Amz-Server-Side-Encryption-Customer-Key":                   struct{}{},
			"X-Amz-Server-Side-Encryption-Customer-Key-Md5":               struct{}{},
			"X-Amz-Storage-Class":                                         struct{}{},
			"X-Amz-Website-Redirect-Location":                             struct{}{},
			"X-Amz-Content-Sha256":                                        struct{}{},
		},
	},
	patterns{"X-Amz-Meta-"},
}

// allowedHoisting is a whitelist for build query headers. The boolean value
// represents whether or not it is a pattern.
var allowedQueryHoisting = inclusiveRules{
	blacklist{requiredSignedHeaders},
	patterns{"X-Amz-"},
}

// Signer applies AWS v4 signing to given request. Use this to sign requests
// that need to be signed with AWS V4 Signatures.
type Signer struct {
	// The authentication credentials the request will be signed against.
	// This value must be set to sign requests.
	Credentials *credentials.Credentials

	// Sets the log level the signer should use when reporting information to
	// the logger. If the logger is nil nothing will be logged. See
	// aws.LogLevelType for more information on available logging levels
	//
	// By default nothing will be logged.
	Debug aws.LogLevelType

	// The logger loging information will be written to. If there the logger
	// is nil, nothing will be logged.
	Logger aws.Logger

	// Disables the Signer's moving HTTP header key/value pairs from the HTTP
	// request header to the request's query string. This is most commonly used
	// with pre-signed requests preventing headers from being added to the
	// request's query string.
	DisableHeaderHoisting bool

	// Disables the automatic escaping of the URI path of the request for the
	// siganture's canonical string's path. For services that do not need additional
	// escaping then use this to disable the signer escaping the path.
	//
	// S3 is an example of a service that does not need additional escaping.
	//
	// http://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html
	DisableURIPathEscaping bool

	// Disales the automatical setting of the HTTP request's Body field with the
	// io.ReadSeeker passed in to the signer. This is useful if you're using a
	// custom wrapper around the body for the io.ReadSeeker and want to preserve
	// the Body value on the Request.Body.
	//
	// This does run the risk of signing a request with a body that will not be
	// sent in the request. Need to ensure that the underlying data of the Body
	// values are the same.
	DisableRequestBodyOverwrite bool

	// currentTimeFn returns the time value which represents the current time.
	// This value should only be used for testing. If it is nil the default
	// time.Now will be used.
	currentTimeFn func() time.Time

	// UnsignedPayload will prevent signing of the payload. This will only
	// work for services that have support for this.
	UnsignedPayload bool
}

// NewSigner returns a Signer pointer configured with the credentials and optional
// option values provided. If not options are provided the Signer will use its
// default configuration.
func NewSigner(credentials *credentials.Credentials, options ...func(*Signer)) *Signer {
	v4 := &Signer{
		Credentials: credentials,
	}

	for _, option := range options {
		option(v4)
	}

	return v4
}

type signingCtx struct {
	ServiceName      string
	Region           string
	Request          *http.Request
	Body             io.ReadSeeker
	Query            url.Values
	Time             time.Time
	ExpireTime       time.Duration
	SignedHeaderVals http.Header

	DisableURIPathEscaping bool

	credValues         credentials.Value
	isPresign          bool
	formattedTime      string
	formattedShortTime string
	unsignedPayload    bool

	bodyDigest       string
	signedHeaders    string
	canonicalHeaders string
	canonicalString  string
	credentialString string
	stringToSign     string
	signature        string
	authorization    string
}

// Sign signs AWS v4 requests with the provided body, service name, region the
// request is made to, and time the request is signed at. The signTime allows
// you to specify that a request is signed for the future, and cannot be
// used until then.
//
// Returns a list of HTTP headers that were included in the signature or an
// error if signing the request failed. Generally for signed requests this value
// is not needed as the full request context will be captured by the http.Request
// value. It is included for reference though.
//
// Sign will set the request's Body to be the `body` parameter passed in. If
// the body is not already an io.ReadCloser, it will be wrapped within one. If
// a `nil` body parameter passed to Sign, the request's Body field will be
// also set to nil. Its important to note that this functionality will not
// change the request's ContentLength of the request.
//
// Sign differs from Presign in that it will sign the request using HTTP
// header values. This type of signing is intended for http.Request values that
// will not be shared, or are shared in a way the header values on the request
// will not be lost.
//
// The requests body is an io.ReadSeeker so the SHA256 of the body can be
// generated. To bypass the signer computing the hash you can set the
// "X-Amz-Content-Sha256" header with a precomputed value. The signer will
// only compute the hash if the request header value is empty.
func (v4 Signer) Sign(r *http.Request, body io.ReadSeeker, service, region string, signTime time.Time) (http.Header, error) {
	return v4.signWithBody(r, body, service, region, 0, false, signTime)
}

// Presign signs AWS v4 requests with the provided body, service name, region
// the request is made to, and time the request is signed at. The signTime
// allows you to specify that a request is signed for the future, and cannot
// be used until then.
//
// Returns a list of HTTP headers that were included in the signature or an
// error if signing the request failed. For presigned requests these headers
// and their values must be included on the HTTP request when it is made. This
// is helpful to know what header values need to be shared with the party the
// presigned request will be distributed to.
//
// Presign differs from Sign in that it will sign the request using query string
// instead of header values. This allows you to share the Presigned Request's
// URL with third parties, or distribute it throughout your system with minimal
// dependencies.
//
// Presign also takes an exp value which is the duration the
// signed request will be valid after the signing time. This is allows you to
// set when the request will expire.
//
// The requests body is an io.ReadSeeker so the SHA256 of the body can be
// generated. To bypass the signer computing the hash you can set the
// "X-Amz-Content-Sha256" header with a precomputed value. The signer will
// only compute the hash if the request header value is empty.
//
// Presigning a S3 request will not compute the body's SHA256 hash by default.
// This is done due to the general use case for S3 presigned URLs is to share
// PUT/GET capabilities. If you would like to include the body's SHA256 in the
// presigned request's signature you can set the "X-Amz-Content-Sha256"
// HTTP header and that will be included in the request's signature.
func (v4 Signer) Presign(r *http.Request, body io.ReadSeeker, service, region string, exp time.Duration, signTime time.Time) (http.Header, error) {
	return v4.signWithBody(r, body, service, region, exp, true, signTime)
}

func (v4 Signer) signWithBody(r *http.Request, body io.ReadSeeker, service, region string, exp time.Duration, isPresign bool, signTime time.Time) (http.Header, error) {
	currentTimeFn := v4.currentTimeFn
	if currentTimeFn == nil {
		currentTimeFn = time.Now
	}

	ctx := &signingCtx{
		Request:                r,
		Body:                   body,
		Query:                  r.URL.Query(),
		Time:                   signTime,
		ExpireTime:             exp,
		isPresign:              isPresign,
		ServiceName:            service,
		Region:                 region,
		DisableURIPathEscaping: v4.DisableURIPathEscaping,
		unsignedPayload:        v4.UnsignedPayload,
	}

	for key := range ctx.Query {
		sort.Strings(ctx.Query[key])
	}

	if ctx.isRequestSigned() {
		ctx.Time = currentTimeFn()
		ctx.handlePresignRemoval()
	}

	var err error
	ctx.credValues, err = v4.Credentials.Get()
	if err != nil {
		return http.Header{}, err
	}

	ctx.sanitizeHostForHeader()
	ctx.assignAmzQueryValues()
	if err := ctx.build(v4.DisableHeaderHoisting); err != nil {
		return nil, err
	}

	// If the request is not presigned the body should be attached to it. This
	// prevents the confusion of wanting to send a signed request without
	// the body the request was signed for attached.
	if !(v4.DisableRequestBodyOverwrite || ctx.isPresign) {
		var reader io.ReadCloser
		if body != nil {
			var ok bool
			if reader, ok = body.(io.ReadCloser); !ok {
				reader = ioutil.NopCloser(body)
			}
		}
		r.Body = reader
	}

	if v4.Debug.Matches(aws.LogDebugWithSigning) {
		v4.logSigningInfo(ctx)
	}

	return ctx.SignedHeaderVals, nil
}

func (ctx *signingCtx) sanitizeHostForHeader() {
	request.SanitizeHostForHeader(ctx.Request)
}

func (ctx *signingCtx) handlePresignRemoval() {
	if !ctx.isPresign {
		return
	}

	// The credentials have expired for this request. The current signing
	// is invalid, and needs to be request because the request will fail.
	ctx.removePresign()

	// Update the request's query string to ensure the values stays in
	// sync in the case retrieving the new credentials fails.
	ctx.Request.URL.RawQuery = ctx.Query.Encode()
}

func (ctx *signingCtx) assignAmzQueryValues() {
	if ctx.isPresign {
		ctx.Query.Set("X-Amz-Algorithm", authHeaderPrefix)
		if ctx.credValues.SessionToken != "" {
			ctx.Query.Set("X-Amz-Security-Token", ctx.credValues.SessionToken)
		} else {
			ctx.Query.Del("X-Amz-Security-Token")
		}

		return
	}

	if ctx.credValues.SessionToken != "" {
		ctx.Request.Header.Set("X-Amz-Security-Token", ctx.credValues.SessionToken)
	}
}

// SignRequestHandler is a named request handler the SDK will use to sign
// service client request with using the V4 signature.
var SignRequestHandler = request.NamedHandler{
	Name: "v4.SignRequestHandler", Fn: SignSDKRequest,
}

// SignSDKRequest signs an AWS request with the V4 signature. This
// request handler should only be used with the SDK's built in service client's
// API operation requests.
//
// This function should not be used on its on its own, but in conjunction with
// an AWS service client's API operation call. To sign a standalone request
// not created by a service client's API operation method use the "Sign" or
// "Presign" functions of the "Signer" type.
//
// If the credentials of the request's config are set to
// credentials.AnonymousCredentials the request will not be signed.
func SignSDKRequest(req *request.Request) {
	signSDKRequestWithCurrTime(req, time.Now)
}

// BuildNamedHandler will build a generic handler for signing.
func BuildNamedHandler(name string, opts ...func(*Signer)) request.NamedHandler {
	return request.NamedHandler{
		Name: name,
		Fn: func(req *request.Request) {
			signSDKRequestWithCurrTime(req, time.Now, opts...)
		},
	}
}

func signSDKRequestWithCurrTime(req *request.Request, curTimeFn func() time.Time, opts ...func(*Signer)) {
	// If the request does not need to be signed ignore the signing of the
	// request if the AnonymousCredentials object is used.
	if req.Config.Credentials == credentials.AnonymousCredentials {
		return
	}

	region := req.ClientInfo.SigningRegion
	if region == "" {
		region = aws.StringValue(req.Config.Region)
	}

	name := req.ClientInfo.SigningName
	if name == "" {
		name = req.ClientInfo.ServiceName
	}

	v4 := NewSigner(req.Config.Credentials, func(v4 *Signer) {
		v4.Debug = req.Config.LogLevel.Value()
		v4.Logger = req.Config.Logger
		v4.DisableHeaderHoisting = req.NotHoist
		v4.currentTimeFn = curTimeFn
		if name == "s3" {
			// S3 service should not have any escaping applied
			v4.DisableURIPathEscaping = true
		}
		// Prevents setting the HTTPRequest's Body. Since the Body could be
		// wrapped in a custom io.Closer that we do not want to be stompped
		// on top of by the signer.
		v4.DisableRequestBodyOverwrite = true
	})

	for _, opt := range opts {
		opt(v4)
	}

	signingTime := req.Time
	if !req.LastSignedAt.IsZero() {
		signingTime = req.LastSignedAt
	}

	signedHeaders, err := v4.signWithBody(req.HTTPRequest, req.GetBody(),
		name, region, req.ExpireTime, req.ExpireTime > 0, signingTime,
	)
	if err != nil {
		req.Error = err
		req.SignedHeaderVals = nil
		return
	}

	req.SignedHeaderVals = signedHeaders
	req.LastSignedAt = curTimeFn()
}

const logSignInfoMsg = `DEBUG: Request Signature:
---[ CANONICAL STRING  ]-----------------------------
%s
---[ STRING TO SIGN ]--------------------------------
%s%s
-----------------------------------------------------`
const logSignedURLMsg = `
---[ SIGNED URL ]------------------------------------
%s`

func (v4 *Signer) logSigningInfo(ctx *signingCtx) {
	signedURLMsg := ""
	if ctx.isPresign {
		signedURLMsg = fmt.Sprintf(logSignedURLMsg, ctx.Request.URL.String())
	}
	msg := fmt.Sprintf(logSignInfoMsg, ctx.canonicalString, ctx.stringToSign, signedURLMsg)
	v4.Logger.Log(msg)
}

func (ctx *signingCtx) build(disableHeaderHoisting bool) error {
	ctx.buildTime()             // no depends
	ctx.buildCredentialString() // no depends

	if err := ctx.buildBodyDigest(); err != nil {
		return err
	}

	unsignedHeaders := ctx.Request.Header
	if ctx.isPresign {
		if !disableHeaderHoisting {
			urlValues := url.Values{}
			urlValues, unsignedHeaders = buildQuery(allowedQueryHoisting, unsignedHeaders) // no depends
			for k := range urlValues {
				ctx.Query[k] = urlValues[k]
			}
		}
	}

	ctx.buildCanonicalHeaders(ignoredHeaders, unsignedHeaders)
	ctx.buildCanonicalString() // depends on canon headers / signed headers
	ctx.buildStringToSign()    // depends on canon string
	ctx.buildSignature()       // depends on string to sign

	if ctx.isPresign {
		ctx.Request.URL.RawQuery += "&X-Amz-Signature=" + ctx.signature
	} else {
		parts := []string{
			authHeaderPrefix + " Credential=" + ctx.credValues.AccessKeyID + "/" + ctx.credentialString,
			"SignedHeaders=" + ctx.signedHeaders,
			"Signature=" + ctx.signature,
		}
		ctx.Request.Header.Set("Authorization", strings.Join(parts, ", "))
	}

	return nil
}

func (ctx *signingCtx) buildTime() {
	ctx.formattedTime = ctx.Time.UTC().Format(timeFormat)
	ctx.formattedShortTime = ctx.Time.UTC().Format(shortTimeFormat)

	if ctx.isPresign {
		duration := int64(ctx.ExpireTime / time.Second)
		ctx.Query.Set("X-Amz-Date", ctx.formattedTime)
		ctx.Query.Set("X-Amz-Expires", strconv.FormatInt(duration, 10))
	} else {
		ctx.Request.Header.Set("X-Amz-Date", ctx.formattedTime)
	}
}

func (ctx *signingCtx) buildCredentialString() {
	ctx.credentialString = strings.Join([]string{
		ctx.formattedShortTime,
		ctx.Region,
		ctx.ServiceName,
		"aws4_request",
	}, "/")

	if ctx.isPresign {
		ctx.Query.Set("X-Amz-Credential", ctx.credValues.AccessKeyID+"/"+ctx.credentialString)
	}
}

func buildQuery(r rule, header http.Header) (url.Values, http.Header) {
	query := url.Values{}
	unsignedHeaders := http.Header{}
	for k, h := range header {
		if r.IsValid(k) {
			query[k] = h
		} else {
			unsignedHeaders[k] = h
		}
	}

	return query, unsignedHeaders
}
func (ctx *signingCtx) buildCanonicalHeaders(r rule, header http.Header) {
	var headers []string
	headers = append(headers, "host")
	for k, v := range header {
		canonicalKey := http.CanonicalHeaderKey(k)
		if !r.IsValid(canonicalKey) {
			continue // ignored header
		}
		if ctx.SignedHeaderVals == nil {
			ctx.SignedHeaderVals = make(http.Header)
		}

		lowerCaseKey := strings.ToLower(k)
		if _, ok := ctx.SignedHeaderVals[lowerCaseKey]; ok {
			// include additional values
			ctx.SignedHeaderVals[lowerCaseKey] = append(ctx.SignedHeaderVals[lowerCaseKey], v...)
			continue
		}

		headers = append(headers, lowerCaseKey)
		ctx.SignedHeaderVals[lowerCaseKey] = v
	}
	sort.Strings(headers)

	ctx.signedHeaders = strings.Join(headers, ";")

	if ctx.isPresign {
		ctx.Query.Set("X-Amz-SignedHeaders", ctx.signedHeaders)
	}

	headerValues := make([]string, len(headers))
	for i, k := range headers {
		if k == "host" {
			if ctx.Request.Host != "" {
				headerValues[i] = "host:" + ctx.Request.Host
			} else {
				headerValues[i] = "host:" + ctx.Request.URL.Host
			}
		} else {
			headerValues[i] = k + ":" +
				strings.Join(ctx.SignedHeaderVals[k], ",")
		}
	}
	stripExcessSpaces(headerValues)
	ctx.canonicalHeaders = strings.Join(headerValues, "\n")
}

func (ctx *signingCtx) buildCanonicalString() {
	ctx.Request.URL.RawQuery = strings.Replace(ctx.Query.Encode(), "+", "%20", -1)

	uri := getURIPath(ctx.Request.URL)

	if !ctx.DisableURIPathEscaping {
		uri = rest.EscapePath(uri, false)
	}

	ctx.canonicalString = strings.Join([]string{
		ctx.Request.Method,
		uri,
		ctx.Request.URL.RawQuery,
		ctx.canonicalHeaders + "\n",
		ctx.signedHeaders,
		ctx.bodyDigest,
	}, "\n")
}

func (ctx *signingCtx) buildStringToSign() {
	ctx.stringToSign = strings.Join([]string{
		authHeaderPrefix,
		ctx.formattedTime,
		ctx.credentialString,
		hex.EncodeToString(makeSha256([]byte(ctx.canonicalString))),
	}, "\n")
}

func (ctx *signingCtx) buildSignature() {
	secret := ctx.credValues.SecretAccessKey
	date := makeHmac([]byte("AWS4"+secret), []byte(ctx.formattedShortTime))
	region := makeHmac(date, []byte(ctx.Region))
	service := makeHmac(region, []byte(ctx.ServiceName))
	credentials := makeHmac(service, []byte("aws4_request"))
	signature := makeHmac(credentials, []byte(ctx.stringToSign))
	ctx.signature = hex.EncodeToString(signature)
}

func (ctx *signingCtx) buildBodyDigest() error {
	hash := ctx.Request.Header.Get("X-Amz-Content-Sha256")
	if hash == "" {
		includeSHA256Header := ctx.unsignedPayload ||
			ctx.ServiceName == "s3" ||
			ctx.ServiceName == "glacier"

		s3Presign := ctx.isPresign && ctx.ServiceName == "s3"

		if ctx.unsignedPayload || s3Presign {
			hash = "UNSIGNED-PAYLOAD"
			includeSHA256Header = !s3Presign
		} else if ctx.Body == nil {
			hash = emptyStringSHA256
		} else {
			if !aws.IsReaderSeekable(ctx.Body) {
				return fmt.Errorf("cannot use unseekable request body %T, for signed request with body", ctx.Body)
			}
			hash = hex.EncodeToString(makeSha256Reader(ctx.Body))
		}

		if includeSHA256Header {
			ctx.Request.Header.Set("X-Amz-Content-Sha256", hash)
		}
	}
	ctx.bodyDigest = hash

	return nil
}

// isRequestSigned returns if the request is currently signed or presigned
func (ctx *signingCtx) isRequestSigned() bool {
	if ctx.isPresign && ctx.Query.Get("X-Amz-Signature") != "" {
		return true
	}
	if ctx.Request.Header.Get("Authorization") != "" {
		return true
	}

	return false
}

// unsign removes signing flags for both signed and presigned requests.
func (ctx *signingCtx) removePresign() {
	ctx.Query.Del("X-Amz-Algorithm")
	ctx.Query.Del("X-Amz-Signature")
	ctx.Query.Del("X-Amz-Security-Token")
	ctx.Query.Del("X-Amz-Date")
	ctx.Query.Del("X-Amz-Expires")
	ctx.Query.Del("X-Amz-Credential")
	ctx.Query.Del("X-Amz-SignedHeaders")
}

func makeHmac(key []byte, data []byte) []byte {
	hash := hmac.New(sha256.New, key)
	hash.Write(data)
	return hash.Sum(nil)
}

func makeSha256(data []byte) []byte {
	hash := sha256.New()
	hash.Write(data)
	return hash.Sum(nil)
}

func makeSha256Reader(reader io.ReadSeeker) []byte {
	hash := sha256.New()
	start, _ := reader.Seek(0, sdkio.SeekCurrent)
	defer reader.Seek(start, sdkio.SeekStart)

	io.Copy(hash, reader)
	return hash.Sum(nil)
}

const doubleSpace = "  "

// stripExcessSpaces will rewrite the passed in slice's string values to not
// contain muliple side-by-side spaces.
func stripExcessSpaces(vals []string) {
	var j, k, l, m, spaces int
	for i, str := range vals {
		// Trim trailing spaces
		for j = len(str) - 1; j >= 0 && str[j] == ' '; j-- {
		}

		// Trim leading spaces
		for k = 0; k < j && str[k] == ' '; k++ {
		}
		str = str[k : j+1]

		// Strip multiple spaces.
		j = strings.Index(str, doubleSpace)
		if j < 0 {
			vals[i] = str
			continue
		}

		buf := []byte(str)
		for k, m, l = j, j, len(buf); k < l; k++ {
			if buf[k] == ' ' {
				if spaces == 0 {
					// First space.
					buf[m] = buf[k]
					m++
				}
				spaces++
			} else {
				// End of multiple spaces.
				spaces = 0
				buf[m] = buf[k]
				m++
			}
		}

		vals[i] = string(buf[:m])
	}
}
