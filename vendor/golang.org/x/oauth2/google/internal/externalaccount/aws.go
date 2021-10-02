// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package externalaccount

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"path"
	"sort"
	"strings"
	"time"

	"golang.org/x/oauth2"
)

type awsSecurityCredentials struct {
	AccessKeyID     string `json:"AccessKeyID"`
	SecretAccessKey string `json:"SecretAccessKey"`
	SecurityToken   string `json:"Token"`
}

// awsRequestSigner is a utility class to sign http requests using a AWS V4 signature.
type awsRequestSigner struct {
	RegionName             string
	AwsSecurityCredentials awsSecurityCredentials
}

// getenv aliases os.Getenv for testing
var getenv = os.Getenv

const (
	// AWS Signature Version 4 signing algorithm identifier.
	awsAlgorithm = "AWS4-HMAC-SHA256"

	// The termination string for the AWS credential scope value as defined in
	// https://docs.aws.amazon.com/general/latest/gr/sigv4-create-string-to-sign.html
	awsRequestType = "aws4_request"

	// The AWS authorization header name for the security session token if available.
	awsSecurityTokenHeader = "x-amz-security-token"

	// The AWS authorization header name for the auto-generated date.
	awsDateHeader = "x-amz-date"

	awsTimeFormatLong  = "20060102T150405Z"
	awsTimeFormatShort = "20060102"
)

func getSha256(input []byte) (string, error) {
	hash := sha256.New()
	if _, err := hash.Write(input); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func getHmacSha256(key, input []byte) ([]byte, error) {
	hash := hmac.New(sha256.New, key)
	if _, err := hash.Write(input); err != nil {
		return nil, err
	}
	return hash.Sum(nil), nil
}

func cloneRequest(r *http.Request) *http.Request {
	r2 := new(http.Request)
	*r2 = *r
	if r.Header != nil {
		r2.Header = make(http.Header, len(r.Header))

		// Find total number of values.
		headerCount := 0
		for _, headerValues := range r.Header {
			headerCount += len(headerValues)
		}
		copiedHeaders := make([]string, headerCount) // shared backing array for headers' values

		for headerKey, headerValues := range r.Header {
			headerCount = copy(copiedHeaders, headerValues)
			r2.Header[headerKey] = copiedHeaders[:headerCount:headerCount]
			copiedHeaders = copiedHeaders[headerCount:]
		}
	}
	return r2
}

func canonicalPath(req *http.Request) string {
	result := req.URL.EscapedPath()
	if result == "" {
		return "/"
	}
	return path.Clean(result)
}

func canonicalQuery(req *http.Request) string {
	queryValues := req.URL.Query()
	for queryKey := range queryValues {
		sort.Strings(queryValues[queryKey])
	}
	return queryValues.Encode()
}

func canonicalHeaders(req *http.Request) (string, string) {
	// Header keys need to be sorted alphabetically.
	var headers []string
	lowerCaseHeaders := make(http.Header)
	for k, v := range req.Header {
		k := strings.ToLower(k)
		if _, ok := lowerCaseHeaders[k]; ok {
			// include additional values
			lowerCaseHeaders[k] = append(lowerCaseHeaders[k], v...)
		} else {
			headers = append(headers, k)
			lowerCaseHeaders[k] = v
		}
	}
	sort.Strings(headers)

	var fullHeaders bytes.Buffer
	for _, header := range headers {
		headerValue := strings.Join(lowerCaseHeaders[header], ",")
		fullHeaders.WriteString(header)
		fullHeaders.WriteRune(':')
		fullHeaders.WriteString(headerValue)
		fullHeaders.WriteRune('\n')
	}

	return strings.Join(headers, ";"), fullHeaders.String()
}

func requestDataHash(req *http.Request) (string, error) {
	var requestData []byte
	if req.Body != nil {
		requestBody, err := req.GetBody()
		if err != nil {
			return "", err
		}
		defer requestBody.Close()

		requestData, err = ioutil.ReadAll(io.LimitReader(requestBody, 1<<20))
		if err != nil {
			return "", err
		}
	}

	return getSha256(requestData)
}

func requestHost(req *http.Request) string {
	if req.Host != "" {
		return req.Host
	}
	return req.URL.Host
}

func canonicalRequest(req *http.Request, canonicalHeaderColumns, canonicalHeaderData string) (string, error) {
	dataHash, err := requestDataHash(req)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%s\n%s\n%s\n%s\n%s\n%s", req.Method, canonicalPath(req), canonicalQuery(req), canonicalHeaderData, canonicalHeaderColumns, dataHash), nil
}

// SignRequest adds the appropriate headers to an http.Request
// or returns an error if something prevented this.
func (rs *awsRequestSigner) SignRequest(req *http.Request) error {
	signedRequest := cloneRequest(req)
	timestamp := now()

	signedRequest.Header.Add("host", requestHost(req))

	if rs.AwsSecurityCredentials.SecurityToken != "" {
		signedRequest.Header.Add(awsSecurityTokenHeader, rs.AwsSecurityCredentials.SecurityToken)
	}

	if signedRequest.Header.Get("date") == "" {
		signedRequest.Header.Add(awsDateHeader, timestamp.Format(awsTimeFormatLong))
	}

	authorizationCode, err := rs.generateAuthentication(signedRequest, timestamp)
	if err != nil {
		return err
	}
	signedRequest.Header.Set("Authorization", authorizationCode)

	req.Header = signedRequest.Header
	return nil
}

func (rs *awsRequestSigner) generateAuthentication(req *http.Request, timestamp time.Time) (string, error) {
	canonicalHeaderColumns, canonicalHeaderData := canonicalHeaders(req)

	dateStamp := timestamp.Format(awsTimeFormatShort)
	serviceName := ""
	if splitHost := strings.Split(requestHost(req), "."); len(splitHost) > 0 {
		serviceName = splitHost[0]
	}

	credentialScope := fmt.Sprintf("%s/%s/%s/%s", dateStamp, rs.RegionName, serviceName, awsRequestType)

	requestString, err := canonicalRequest(req, canonicalHeaderColumns, canonicalHeaderData)
	if err != nil {
		return "", err
	}
	requestHash, err := getSha256([]byte(requestString))
	if err != nil {
		return "", err
	}

	stringToSign := fmt.Sprintf("%s\n%s\n%s\n%s", awsAlgorithm, timestamp.Format(awsTimeFormatLong), credentialScope, requestHash)

	signingKey := []byte("AWS4" + rs.AwsSecurityCredentials.SecretAccessKey)
	for _, signingInput := range []string{
		dateStamp, rs.RegionName, serviceName, awsRequestType, stringToSign,
	} {
		signingKey, err = getHmacSha256(signingKey, []byte(signingInput))
		if err != nil {
			return "", err
		}
	}

	return fmt.Sprintf("%s Credential=%s/%s, SignedHeaders=%s, Signature=%s", awsAlgorithm, rs.AwsSecurityCredentials.AccessKeyID, credentialScope, canonicalHeaderColumns, hex.EncodeToString(signingKey)), nil
}

type awsCredentialSource struct {
	EnvironmentID               string
	RegionURL                   string
	RegionalCredVerificationURL string
	CredVerificationURL         string
	TargetResource              string
	requestSigner               *awsRequestSigner
	region                      string
	ctx                         context.Context
	client                      *http.Client
}

type awsRequestHeader struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

type awsRequest struct {
	URL     string             `json:"url"`
	Method  string             `json:"method"`
	Headers []awsRequestHeader `json:"headers"`
}

func (cs awsCredentialSource) doRequest(req *http.Request) (*http.Response, error) {
	if cs.client == nil {
		cs.client = oauth2.NewClient(cs.ctx, nil)
	}
	return cs.client.Do(req.WithContext(cs.ctx))
}

func (cs awsCredentialSource) subjectToken() (string, error) {
	if cs.requestSigner == nil {
		awsSecurityCredentials, err := cs.getSecurityCredentials()
		if err != nil {
			return "", err
		}

		if cs.region, err = cs.getRegion(); err != nil {
			return "", err
		}

		cs.requestSigner = &awsRequestSigner{
			RegionName:             cs.region,
			AwsSecurityCredentials: awsSecurityCredentials,
		}
	}

	// Generate the signed request to AWS STS GetCallerIdentity API.
	// Use the required regional endpoint. Otherwise, the request will fail.
	req, err := http.NewRequest("POST", strings.Replace(cs.RegionalCredVerificationURL, "{region}", cs.region, 1), nil)
	if err != nil {
		return "", err
	}
	// The full, canonical resource name of the workload identity pool
	// provider, with or without the HTTPS prefix.
	// Including this header as part of the signature is recommended to
	// ensure data integrity.
	if cs.TargetResource != "" {
		req.Header.Add("x-goog-cloud-target-resource", cs.TargetResource)
	}
	cs.requestSigner.SignRequest(req)

	/*
	   The GCP STS endpoint expects the headers to be formatted as:
	   # [
	   #   {key: 'x-amz-date', value: '...'},
	   #   {key: 'Authorization', value: '...'},
	   #   ...
	   # ]
	   # And then serialized as:
	   # quote(json.dumps({
	   #   url: '...',
	   #   method: 'POST',
	   #   headers: [{key: 'x-amz-date', value: '...'}, ...]
	   # }))
	*/

	awsSignedReq := awsRequest{
		URL:    req.URL.String(),
		Method: "POST",
	}
	for headerKey, headerList := range req.Header {
		for _, headerValue := range headerList {
			awsSignedReq.Headers = append(awsSignedReq.Headers, awsRequestHeader{
				Key:   headerKey,
				Value: headerValue,
			})
		}
	}
	sort.Slice(awsSignedReq.Headers, func(i, j int) bool {
		headerCompare := strings.Compare(awsSignedReq.Headers[i].Key, awsSignedReq.Headers[j].Key)
		if headerCompare == 0 {
			return strings.Compare(awsSignedReq.Headers[i].Value, awsSignedReq.Headers[j].Value) < 0
		}
		return headerCompare < 0
	})

	result, err := json.Marshal(awsSignedReq)
	if err != nil {
		return "", err
	}
	return url.QueryEscape(string(result)), nil
}

func (cs *awsCredentialSource) getRegion() (string, error) {
	if envAwsRegion := getenv("AWS_REGION"); envAwsRegion != "" {
		return envAwsRegion, nil
	}
	if envAwsRegion := getenv("AWS_DEFAULT_REGION"); envAwsRegion != "" {
		return envAwsRegion, nil
	}

	if cs.RegionURL == "" {
		return "", errors.New("oauth2/google: unable to determine AWS region")
	}

	req, err := http.NewRequest("GET", cs.RegionURL, nil)
	if err != nil {
		return "", err
	}

	resp, err := cs.doRequest(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	respBody, err := ioutil.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", err
	}

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("oauth2/google: unable to retrieve AWS region - %s", string(respBody))
	}

	// This endpoint will return the region in format: us-east-2b.
	// Only the us-east-2 part should be used.
	respBodyEnd := 0
	if len(respBody) > 1 {
		respBodyEnd = len(respBody) - 1
	}
	return string(respBody[:respBodyEnd]), nil
}

func (cs *awsCredentialSource) getSecurityCredentials() (result awsSecurityCredentials, err error) {
	if accessKeyID := getenv("AWS_ACCESS_KEY_ID"); accessKeyID != "" {
		if secretAccessKey := getenv("AWS_SECRET_ACCESS_KEY"); secretAccessKey != "" {
			return awsSecurityCredentials{
				AccessKeyID:     accessKeyID,
				SecretAccessKey: secretAccessKey,
				SecurityToken:   getenv("AWS_SESSION_TOKEN"),
			}, nil
		}
	}

	roleName, err := cs.getMetadataRoleName()
	if err != nil {
		return
	}

	credentials, err := cs.getMetadataSecurityCredentials(roleName)
	if err != nil {
		return
	}

	if credentials.AccessKeyID == "" {
		return result, errors.New("oauth2/google: missing AccessKeyId credential")
	}

	if credentials.SecretAccessKey == "" {
		return result, errors.New("oauth2/google: missing SecretAccessKey credential")
	}

	return credentials, nil
}

func (cs *awsCredentialSource) getMetadataSecurityCredentials(roleName string) (awsSecurityCredentials, error) {
	var result awsSecurityCredentials

	req, err := http.NewRequest("GET", fmt.Sprintf("%s/%s", cs.CredVerificationURL, roleName), nil)
	if err != nil {
		return result, err
	}
	req.Header.Add("Content-Type", "application/json")

	resp, err := cs.doRequest(req)
	if err != nil {
		return result, err
	}
	defer resp.Body.Close()

	respBody, err := ioutil.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return result, err
	}

	if resp.StatusCode != 200 {
		return result, fmt.Errorf("oauth2/google: unable to retrieve AWS security credentials - %s", string(respBody))
	}

	err = json.Unmarshal(respBody, &result)
	return result, err
}

func (cs *awsCredentialSource) getMetadataRoleName() (string, error) {
	if cs.CredVerificationURL == "" {
		return "", errors.New("oauth2/google: unable to determine the AWS metadata server security credentials endpoint")
	}

	req, err := http.NewRequest("GET", cs.CredVerificationURL, nil)
	if err != nil {
		return "", err
	}

	resp, err := cs.doRequest(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	respBody, err := ioutil.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", err
	}

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("oauth2/google: unable to retrieve AWS role name - %s", string(respBody))
	}

	return string(respBody), nil
}
