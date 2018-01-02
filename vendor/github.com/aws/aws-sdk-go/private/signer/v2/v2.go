package v2

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
)

var (
	errInvalidMethod = errors.New("v2 signer only handles HTTP POST")
)

const (
	signatureVersion = "2"
	signatureMethod  = "HmacSHA256"
	timeFormat       = "2006-01-02T15:04:05Z"
)

type signer struct {
	// Values that must be populated from the request
	Request     *http.Request
	Time        time.Time
	Credentials *credentials.Credentials
	Debug       aws.LogLevelType
	Logger      aws.Logger

	Query        url.Values
	stringToSign string
	signature    string
}

// SignRequestHandler is a named request handler the SDK will use to sign
// service client request with using the V4 signature.
var SignRequestHandler = request.NamedHandler{
	Name: "v2.SignRequestHandler", Fn: SignSDKRequest,
}

// SignSDKRequest requests with signature version 2.
//
// Will sign the requests with the service config's Credentials object
// Signing is skipped if the credentials is the credentials.AnonymousCredentials
// object.
func SignSDKRequest(req *request.Request) {
	// If the request does not need to be signed ignore the signing of the
	// request if the AnonymousCredentials object is used.
	if req.Config.Credentials == credentials.AnonymousCredentials {
		return
	}

	if req.HTTPRequest.Method != "POST" && req.HTTPRequest.Method != "GET" {
		// The V2 signer only supports GET and POST
		req.Error = errInvalidMethod
		return
	}

	v2 := signer{
		Request:     req.HTTPRequest,
		Time:        req.Time,
		Credentials: req.Config.Credentials,
		Debug:       req.Config.LogLevel.Value(),
		Logger:      req.Config.Logger,
	}

	req.Error = v2.Sign()

	if req.Error != nil {
		return
	}

	if req.HTTPRequest.Method == "POST" {
		// Set the body of the request based on the modified query parameters
		req.SetStringBody(v2.Query.Encode())

		// Now that the body has changed, remove any Content-Length header,
		// because it will be incorrect
		req.HTTPRequest.ContentLength = 0
		req.HTTPRequest.Header.Del("Content-Length")
	} else {
		req.HTTPRequest.URL.RawQuery = v2.Query.Encode()
	}
}

func (v2 *signer) Sign() error {
	credValue, err := v2.Credentials.Get()
	if err != nil {
		return err
	}

	if v2.Request.Method == "POST" {
		// Parse the HTTP request to obtain the query parameters that will
		// be used to build the string to sign. Note that because the HTTP
		// request will need to be modified, the PostForm and Form properties
		// are reset to nil after parsing.
		v2.Request.ParseForm()
		v2.Query = v2.Request.PostForm
		v2.Request.PostForm = nil
		v2.Request.Form = nil
	} else {
		v2.Query = v2.Request.URL.Query()
	}

	// Set new query parameters
	v2.Query.Set("AWSAccessKeyId", credValue.AccessKeyID)
	v2.Query.Set("SignatureVersion", signatureVersion)
	v2.Query.Set("SignatureMethod", signatureMethod)
	v2.Query.Set("Timestamp", v2.Time.UTC().Format(timeFormat))
	if credValue.SessionToken != "" {
		v2.Query.Set("SecurityToken", credValue.SessionToken)
	}

	// in case this is a retry, ensure no signature present
	v2.Query.Del("Signature")

	method := v2.Request.Method
	host := v2.Request.URL.Host
	path := v2.Request.URL.Path
	if path == "" {
		path = "/"
	}

	// obtain all of the query keys and sort them
	queryKeys := make([]string, 0, len(v2.Query))
	for key := range v2.Query {
		queryKeys = append(queryKeys, key)
	}
	sort.Strings(queryKeys)

	// build URL-encoded query keys and values
	queryKeysAndValues := make([]string, len(queryKeys))
	for i, key := range queryKeys {
		k := strings.Replace(url.QueryEscape(key), "+", "%20", -1)
		v := strings.Replace(url.QueryEscape(v2.Query.Get(key)), "+", "%20", -1)
		queryKeysAndValues[i] = k + "=" + v
	}

	// join into one query string
	query := strings.Join(queryKeysAndValues, "&")

	// build the canonical string for the V2 signature
	v2.stringToSign = strings.Join([]string{
		method,
		host,
		path,
		query,
	}, "\n")

	hash := hmac.New(sha256.New, []byte(credValue.SecretAccessKey))
	hash.Write([]byte(v2.stringToSign))
	v2.signature = base64.StdEncoding.EncodeToString(hash.Sum(nil))
	v2.Query.Set("Signature", v2.signature)

	if v2.Debug.Matches(aws.LogDebugWithSigning) {
		v2.logSigningInfo()
	}

	return nil
}

const logSignInfoMsg = `DEBUG: Request Signature:
---[ STRING TO SIGN ]--------------------------------
%s
---[ SIGNATURE ]-------------------------------------
%s
-----------------------------------------------------`

func (v2 *signer) logSigningInfo() {
	msg := fmt.Sprintf(logSignInfoMsg, v2.stringToSign, v2.Query.Get("Signature"))
	v2.Logger.Log(msg)
}
