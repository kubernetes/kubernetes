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
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type restClient struct {
	httpClient *http.Client
	logger     *log.Logger
}

type request struct {
	Method      string
	URL         string
	ContentType string
	Body        io.Reader
	Token       string
}

type page struct {
	Items            []interface{} `json:"items"`
	NextPageLink     string        `json:"nextPageLink"`
	PreviousPageLink string        `json:"previousPageLink"`
}

type documentList struct {
	Items []interface{}
}

const appJson string = "application/json"

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

func (client *restClient) Get(url string, token string) (res *http.Response, err error) {
	req := request{"GET", url, "", nil, token}
	res, err = client.Do(&req)
	return
}

func (client *restClient) GetList(endpoint string, url string, token string) (result []byte, err error) {
	req := request{"GET", url, "", nil, token}
	res, err := client.Do(&req)
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
		req = request{"GET", endpoint + page.NextPageLink, "", nil, token}
		res, err = client.Do(&req)
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

func (client *restClient) Post(url string, contentType string, body io.Reader, token string) (res *http.Response, err error) {
	if contentType == "" {
		contentType = appJson
	}

	req := request{"POST", url, contentType, body, token}
	res, err = client.Do(&req)
	return
}

func (client *restClient) Delete(url string, token string) (res *http.Response, err error) {
	req := request{"DELETE", url, "", nil, token}
	res, err = client.Do(&req)
	return
}

func (client *restClient) Do(req *request) (res *http.Response, err error) {
	r, err := http.NewRequest(req.Method, req.URL, req.Body)
	if err != nil {
		client.logger.Printf("An error occured creating request %s on %s. Error: %s", req.Method, req.URL, err)
		return
	}
	if req.ContentType != "" {
		r.Header.Add("Content-Type", req.ContentType)
	}
	if req.Token != "" {
		r.Header.Add("Authorization", "Bearer "+req.Token)
	}
	res, err = client.httpClient.Do(r)
	if err != nil {
		client.logger.Printf("An error occured when calling %s on %s. Error: %s", req.Method, req.URL, err)
		return
	}

	client.logger.Printf("[%s] %s - %s %s", res.Header.Get("request-id"), res.Status, req.Method, req.URL)
	return
}

func (client *restClient) MultipartUploadFile(url, filePath string, params map[string]string, token string) (res *http.Response, err error) {
	file, err := os.Open(filePath)
	if err != nil {
		return
	}
	defer file.Close()
	return client.MultipartUpload(url, file, filepath.Base(filePath), params, token)
}

func (client *restClient) MultipartUpload(url string, reader io.Reader, filename string, params map[string]string, token string) (res *http.Response, err error) {
	// The mime/multipart package does not support streaming multipart data from disk,
	// at least not without complicated, problematic goroutines that simultaneously read/write into a buffer.
	// A much easier approach is to just construct the multipart request by hand, using io.MultiPart to
	// concatenate the parts of the request together into a single io.Reader.
	boundary := client.randomBoundary()
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

	res, err = client.Do(&request{"POST", url, contentType, io.MultiReader(parts...), token})

	return
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
