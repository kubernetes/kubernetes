// Copyright (c) 2016 VMware, Inc. All Rights Reserved.
//
// This product is licensed to you under the Apache License, Version 2.0 (the "License").
// You may not use this product except in compliance with the License.
//
// This product may include a number of subcomponents with separate copyright notices and
// license terms. Your use of these subcomponents is subject to the terms and conditions
// of the subcomponent's license, as noted in the LICENSE file.

package photon

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type restClient struct {
	httpClient                *http.Client
	logger                    *log.Logger
	Auth                      *AuthAPI
	UpdateAccessTokenCallback TokenCallback
}

type request struct {
	Method      string
	URL         string
	ContentType string
	Body        io.Reader
	Tokens      *TokenOptions
}

type page struct {
	Items            []interface{} `json:"items"`
	NextPageLink     string        `json:"nextPageLink"`
	PreviousPageLink string        `json:"previousPageLink"`
}

type documentList struct {
	Items []interface{}
}

type bodyRewinder func() io.Reader

const appJson string = "application/json"

// Root URL specifies the API version.
const rootUrl string = "/v1"

// From https://golang.org/src/mime/multipart/writer.go
var quoteEscaper = strings.NewReplacer("\\", "\\\\", `"`, "\\\"")

func (client *restClient) AppendSlice(origSlice []interface{}, dataToAppend []interface{}) []interface{} {
	origLen := len(origSlice)
	newLen := origLen + len(dataToAppend)

	if newLen > cap(origSlice) {
		newSlice := make([]interface{}, (newLen+1)*2)
		copy(newSlice, origSlice)
		origSlice = newSlice
	}

	origSlice = origSlice[0:newLen]
	copy(origSlice[origLen:newLen], dataToAppend)

	return origSlice
}

func (client *restClient) Get(url string, tokens *TokenOptions) (res *http.Response, err error) {
	req := request{"GET", url, "", nil, tokens}
	res, err = client.SendRequest(&req, nil)
	return
}

func (client *restClient) GetList(endpoint string, url string, tokens *TokenOptions) (result []byte, err error) {
	req := request{"GET", url, "", nil, tokens}
	res, err := client.SendRequest(&req, nil)
	if err != nil {
		return
	}
	res, err = getError(res)
	if err != nil {
		return
	}

	decoder := json.NewDecoder(res.Body)
	decoder.UseNumber()

	page := &page{}
	err = decoder.Decode(page)
	if err != nil {
		return
	}

	documentList := &documentList{}
	documentList.Items = client.AppendSlice(documentList.Items, page.Items)

	for page.NextPageLink != "" {
		req = request{"GET", endpoint + page.NextPageLink, "", nil, tokens}
		res, err = client.SendRequest(&req, nil)
		if err != nil {
			return
		}
		res, err = getError(res)
		if err != nil {
			return
		}

		decoder = json.NewDecoder(res.Body)
		decoder.UseNumber()

		page.NextPageLink = ""
		page.PreviousPageLink = ""

		err = decoder.Decode(page)
		if err != nil {
			return
		}

		documentList.Items = client.AppendSlice(documentList.Items, page.Items)
	}

	result, err = json.Marshal(documentList)

	return
}

func (client *restClient) Post(url string, contentType string, body io.ReadSeeker, tokens *TokenOptions) (res *http.Response, err error) {
	if contentType == "" {
		contentType = appJson
	}

	req := request{"POST", url, contentType, body, tokens}
	rewinder := func() io.Reader {
		body.Seek(0, 0)
		return body
	}
	res, err = client.SendRequest(&req, rewinder)
	return
}

func (client *restClient) Patch(url string, contentType string, body io.ReadSeeker, tokens *TokenOptions) (res *http.Response, err error) {
	if contentType == "" {
		contentType = appJson
	}

	req := request{"PATCH", url, contentType, body, tokens}
	rewinder := func() io.Reader {
		body.Seek(0, 0)
		return body
	}
	res, err = client.SendRequest(&req, rewinder)
	return
}

func (client *restClient) Put(url string, contentType string, body io.ReadSeeker, tokens *TokenOptions) (res *http.Response, err error) {
	if contentType == "" {
		contentType = appJson
	}

	req := request{"PUT", url, contentType, body, tokens}
	rewinder := func() io.Reader {
		body.Seek(0, 0)
		return body
	}
	res, err = client.SendRequest(&req, rewinder)
	return
}

func (client *restClient) Delete(url string, tokens *TokenOptions) (res *http.Response, err error) {
	req := request{"DELETE", url, "", nil, tokens}
	res, err = client.SendRequest(&req, nil)
	return
}

func (client *restClient) SendRequest(req *request, bodyRewinder bodyRewinder) (res *http.Response, err error) {
	res, err = client.sendRequestHelper(req)
	// In most cases, we'll return immediately
	// If the operation succeeded, but we got a 401 response and if we're using
	// authentication, then we'll look into the body to see if the token expired
	if err != nil {
		return res, err
	}
	if res.StatusCode != 401 {
		// It's not a 401, so the token didn't expire
		return res, err
	}
	if req.Tokens == nil || req.Tokens.AccessToken == "" {
		// We don't have a token, so we can't renew the token, no need to proceed
		return res, err
	}

	// We're going to look in the body to see if it failed because the token expired
	// This means we need to read the body, but the functions that call us also
	// expect to read the body. So we read the body, then create a new reader
	// so they can read the body as normal.
	body, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return res, err
	}
	res.Body = ioutil.NopCloser(bytes.NewReader(body))

	// Now see if we had an expired token or not
	var apiError ApiError
	err = json.Unmarshal(body, &apiError)
	if err != nil {
		return res, err
	}
	if apiError.Code != "ExpiredAuthToken" {
		return res, nil
	}

	// We were told that the access token expired, so try to renew it.
	// Note that this looks recursive because GetTokensByRefreshToken() will
	// call the /auth API, and therefore SendRequest(). However, it calls
	// without a token, so we avoid having a loop
	newTokens, err := client.Auth.GetTokensByRefreshToken(req.Tokens.RefreshToken)
	if err != nil {
		return res, err
	}
	req.Tokens.AccessToken = newTokens.AccessToken
	if client.UpdateAccessTokenCallback != nil {
		client.UpdateAccessTokenCallback(newTokens.AccessToken)
	}
	if req.Body != nil && bodyRewinder != nil {
		req.Body = bodyRewinder()
	}
	res, err = client.sendRequestHelper(req)
	return res, nil
}

func (client *restClient) sendRequestHelper(req *request) (res *http.Response, err error) {
	r, err := http.NewRequest(req.Method, req.URL, req.Body)
	if err != nil {
		client.logger.Printf("An error occured creating request %s on %s. Error: %s", req.Method, req.URL, err)
		return
	}
	if req.ContentType != "" {
		r.Header.Add("Content-Type", req.ContentType)
	}
	if req.Tokens != nil && req.Tokens.AccessToken != "" {
		r.Header.Add("Authorization", "Bearer "+req.Tokens.AccessToken)
	}
	res, err = client.httpClient.Do(r)
	if err != nil {
		client.logger.Printf("An error occured when calling %s on %s. Error: %s", req.Method, req.URL, err)
		return
	}

	client.logger.Printf("[%s] %s - %s %s", res.Header.Get("request-id"), res.Status, req.Method, req.URL)
	return
}

func (client *restClient) MultipartUploadFile(url, filePath string, params map[string]string, tokens *TokenOptions) (res *http.Response, err error) {
	file, err := os.Open(filePath)
	if err != nil {
		return
	}
	defer file.Close()
	return client.MultipartUpload(url, file, filepath.Base(filePath), params, tokens)
}

func (client *restClient) MultipartUpload(url string, reader io.ReadSeeker, filename string, params map[string]string, tokens *TokenOptions) (res *http.Response, err error) {
	boundary := client.randomBoundary()
	multiReader, contentType := client.createMultiReader(reader, filename, params, boundary)
	rewinder := func() io.Reader {
		reader.Seek(0, 0)
		multiReader, _ := client.createMultiReader(reader, filename, params, boundary)
		return multiReader
	}
	res, err = client.SendRequest(&request{"POST", url, contentType, multiReader, tokens}, rewinder)

	return
}

func (client *restClient) createMultiReader(reader io.ReadSeeker, filename string, params map[string]string, boundary string) (io.Reader, string) {
	// The mime/multipart package does not support streaming multipart data from disk,
	// at least not without complicated, problematic goroutines that simultaneously read/write into a buffer.
	// A much easier approach is to just construct the multipart request by hand, using io.MultiPart to
	// concatenate the parts of the request together into a single io.Reader.
	parts := []io.Reader{}

	// Create a part for each key, val pair in params
	for k, v := range params {
		parts = append(parts, client.createFieldPart(k, v, boundary))
	}

	start := fmt.Sprintf("\r\n--%s\r\n", boundary)
	start += fmt.Sprintf("Content-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n", quoteEscaper.Replace(filename))
	start += fmt.Sprintf("Content-Type: application/octet-stream\r\n\r\n")
	end := fmt.Sprintf("\r\n--%s--", boundary)

	// The request will consist of a reader to begin the request, a reader which points
	// to the file data on disk, and a reader containing the closing boundary of the request.
	parts = append(parts, strings.NewReader(start), reader, strings.NewReader(end))

	contentType := fmt.Sprintf("multipart/form-data; boundary=%s", boundary)

	return io.MultiReader(parts...), contentType
}

// From https://golang.org/src/mime/multipart/writer.go
func (client *restClient) randomBoundary() string {
	var buf [30]byte
	_, err := io.ReadFull(rand.Reader, buf[:])
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf("%x", buf[:])
}

// Creates a reader that encapsulates a single multipart form part
func (client *restClient) createFieldPart(fieldname, value, boundary string) io.Reader {
	str := fmt.Sprintf("\r\n--%s\r\n", boundary)
	str += fmt.Sprintf("Content-Disposition: form-data; name=\"%s\"\r\n\r\n", quoteEscaper.Replace(fieldname))
	str += value
	return strings.NewReader(str)
}
