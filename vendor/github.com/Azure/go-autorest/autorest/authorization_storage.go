package autorest

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

// SharedKeyType defines the enumeration for the various shared key types.
// See https://docs.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key for details on the shared key types.
type SharedKeyType string

const (
	// SharedKey is used to authorize against blobs, files and queues services.
	SharedKey SharedKeyType = "sharedKey"

	// SharedKeyForTable is used to authorize against the table service.
	SharedKeyForTable SharedKeyType = "sharedKeyTable"

	// SharedKeyLite is used to authorize against blobs, files and queues services.  It's provided for
	// backwards compatibility with API versions before 2009-09-19.  Prefer SharedKey instead.
	SharedKeyLite SharedKeyType = "sharedKeyLite"

	// SharedKeyLiteForTable is used to authorize against the table service.  It's provided for
	// backwards compatibility with older table API versions.  Prefer SharedKeyForTable instead.
	SharedKeyLiteForTable SharedKeyType = "sharedKeyLiteTable"
)

const (
	headerAccept            = "Accept"
	headerAcceptCharset     = "Accept-Charset"
	headerContentEncoding   = "Content-Encoding"
	headerContentLength     = "Content-Length"
	headerContentMD5        = "Content-MD5"
	headerContentLanguage   = "Content-Language"
	headerIfModifiedSince   = "If-Modified-Since"
	headerIfMatch           = "If-Match"
	headerIfNoneMatch       = "If-None-Match"
	headerIfUnmodifiedSince = "If-Unmodified-Since"
	headerDate              = "Date"
	headerXMSDate           = "X-Ms-Date"
	headerXMSVersion        = "x-ms-version"
	headerRange             = "Range"
)

const storageEmulatorAccountName = "devstoreaccount1"

// SharedKeyAuthorizer implements an authorization for Shared Key
// this can be used for interaction with Blob, File and Queue Storage Endpoints
type SharedKeyAuthorizer struct {
	accountName string
	accountKey  []byte
	keyType     SharedKeyType
}

// NewSharedKeyAuthorizer creates a SharedKeyAuthorizer using the provided credentials and shared key type.
func NewSharedKeyAuthorizer(accountName, accountKey string, keyType SharedKeyType) (*SharedKeyAuthorizer, error) {
	key, err := base64.StdEncoding.DecodeString(accountKey)
	if err != nil {
		return nil, fmt.Errorf("malformed storage account key: %v", err)
	}
	return &SharedKeyAuthorizer{
		accountName: accountName,
		accountKey:  key,
		keyType:     keyType,
	}, nil
}

// WithAuthorization returns a PrepareDecorator that adds an HTTP Authorization header whose
// value is "<SharedKeyType> " followed by the computed key.
// This can be used for the Blob, Queue, and File Services
//
// from: https://docs.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key
// You may use Shared Key authorization to authorize a request made against the
// 2009-09-19 version and later of the Blob and Queue services,
// and version 2014-02-14 and later of the File services.
func (sk *SharedKeyAuthorizer) WithAuthorization() PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			r, err := p.Prepare(r)
			if err != nil {
				return r, err
			}

			sk, err := buildSharedKey(sk.accountName, sk.accountKey, r, sk.keyType)
			if err != nil {
				return r, err
			}
			return Prepare(r, WithHeader(headerAuthorization, sk))
		})
	}
}

func buildSharedKey(accName string, accKey []byte, req *http.Request, keyType SharedKeyType) (string, error) {
	canRes, err := buildCanonicalizedResource(accName, req.URL.String(), keyType)
	if err != nil {
		return "", err
	}

	if req.Header == nil {
		req.Header = http.Header{}
	}

	// ensure date is set
	if req.Header.Get(headerDate) == "" && req.Header.Get(headerXMSDate) == "" {
		date := time.Now().UTC().Format(http.TimeFormat)
		req.Header.Set(headerXMSDate, date)
	}
	canString, err := buildCanonicalizedString(req.Method, req.Header, canRes, keyType)
	if err != nil {
		return "", err
	}
	return createAuthorizationHeader(accName, accKey, canString, keyType), nil
}

func buildCanonicalizedResource(accountName, uri string, keyType SharedKeyType) (string, error) {
	errMsg := "buildCanonicalizedResource error: %s"
	u, err := url.Parse(uri)
	if err != nil {
		return "", fmt.Errorf(errMsg, err.Error())
	}

	cr := bytes.NewBufferString("")
	if accountName != storageEmulatorAccountName {
		cr.WriteString("/")
		cr.WriteString(getCanonicalizedAccountName(accountName))
	}

	if len(u.Path) > 0 {
		// Any portion of the CanonicalizedResource string that is derived from
		// the resource's URI should be encoded exactly as it is in the URI.
		// -- https://msdn.microsoft.com/en-gb/library/azure/dd179428.aspx
		cr.WriteString(u.EscapedPath())
	} else {
		// a slash is required to indicate the root path
		cr.WriteString("/")
	}

	params, err := url.ParseQuery(u.RawQuery)
	if err != nil {
		return "", fmt.Errorf(errMsg, err.Error())
	}

	// See https://github.com/Azure/azure-storage-net/blob/master/Lib/Common/Core/Util/AuthenticationUtility.cs#L277
	if keyType == SharedKey {
		if len(params) > 0 {
			cr.WriteString("\n")

			keys := []string{}
			for key := range params {
				keys = append(keys, key)
			}
			sort.Strings(keys)

			completeParams := []string{}
			for _, key := range keys {
				if len(params[key]) > 1 {
					sort.Strings(params[key])
				}

				completeParams = append(completeParams, fmt.Sprintf("%s:%s", key, strings.Join(params[key], ",")))
			}
			cr.WriteString(strings.Join(completeParams, "\n"))
		}
	} else {
		// search for "comp" parameter, if exists then add it to canonicalizedresource
		if v, ok := params["comp"]; ok {
			cr.WriteString("?comp=" + v[0])
		}
	}

	return string(cr.Bytes()), nil
}

func getCanonicalizedAccountName(accountName string) string {
	// since we may be trying to access a secondary storage account, we need to
	// remove the -secondary part of the storage name
	return strings.TrimSuffix(accountName, "-secondary")
}

func buildCanonicalizedString(verb string, headers http.Header, canonicalizedResource string, keyType SharedKeyType) (string, error) {
	contentLength := headers.Get(headerContentLength)
	if contentLength == "0" {
		contentLength = ""
	}
	date := headers.Get(headerDate)
	if v := headers.Get(headerXMSDate); v != "" {
		if keyType == SharedKey || keyType == SharedKeyLite {
			date = ""
		} else {
			date = v
		}
	}
	var canString string
	switch keyType {
	case SharedKey:
		canString = strings.Join([]string{
			verb,
			headers.Get(headerContentEncoding),
			headers.Get(headerContentLanguage),
			contentLength,
			headers.Get(headerContentMD5),
			headers.Get(headerContentType),
			date,
			headers.Get(headerIfModifiedSince),
			headers.Get(headerIfMatch),
			headers.Get(headerIfNoneMatch),
			headers.Get(headerIfUnmodifiedSince),
			headers.Get(headerRange),
			buildCanonicalizedHeader(headers),
			canonicalizedResource,
		}, "\n")
	case SharedKeyForTable:
		canString = strings.Join([]string{
			verb,
			headers.Get(headerContentMD5),
			headers.Get(headerContentType),
			date,
			canonicalizedResource,
		}, "\n")
	case SharedKeyLite:
		canString = strings.Join([]string{
			verb,
			headers.Get(headerContentMD5),
			headers.Get(headerContentType),
			date,
			buildCanonicalizedHeader(headers),
			canonicalizedResource,
		}, "\n")
	case SharedKeyLiteForTable:
		canString = strings.Join([]string{
			date,
			canonicalizedResource,
		}, "\n")
	default:
		return "", fmt.Errorf("key type '%s' is not supported", keyType)
	}
	return canString, nil
}

func buildCanonicalizedHeader(headers http.Header) string {
	cm := make(map[string]string)

	for k := range headers {
		headerName := strings.TrimSpace(strings.ToLower(k))
		if strings.HasPrefix(headerName, "x-ms-") {
			cm[headerName] = headers.Get(k)
		}
	}

	if len(cm) == 0 {
		return ""
	}

	keys := []string{}
	for key := range cm {
		keys = append(keys, key)
	}

	sort.Strings(keys)

	ch := bytes.NewBufferString("")

	for _, key := range keys {
		ch.WriteString(key)
		ch.WriteRune(':')
		ch.WriteString(cm[key])
		ch.WriteRune('\n')
	}

	return strings.TrimSuffix(string(ch.Bytes()), "\n")
}

func createAuthorizationHeader(accountName string, accountKey []byte, canonicalizedString string, keyType SharedKeyType) string {
	h := hmac.New(sha256.New, accountKey)
	h.Write([]byte(canonicalizedString))
	signature := base64.StdEncoding.EncodeToString(h.Sum(nil))
	var key string
	switch keyType {
	case SharedKey, SharedKeyForTable:
		key = "SharedKey"
	case SharedKeyLite, SharedKeyLiteForTable:
		key = "SharedKeyLite"
	}
	return fmt.Sprintf("%s %s:%s", key, getCanonicalizedAccountName(accountName), signature)
}
