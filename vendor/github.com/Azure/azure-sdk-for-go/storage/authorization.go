// Package storage provides clients for Microsoft Azure Storage Services.
package storage

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
	"fmt"
	"net/url"
	"sort"
	"strings"
)

// See: https://docs.microsoft.com/rest/api/storageservices/fileservices/authentication-for-the-azure-storage-services

type authentication string

const (
	sharedKey             authentication = "sharedKey"
	sharedKeyForTable     authentication = "sharedKeyTable"
	sharedKeyLite         authentication = "sharedKeyLite"
	sharedKeyLiteForTable authentication = "sharedKeyLiteTable"

	// headers
	headerAcceptCharset           = "Accept-Charset"
	headerAuthorization           = "Authorization"
	headerContentLength           = "Content-Length"
	headerDate                    = "Date"
	headerXmsDate                 = "x-ms-date"
	headerXmsVersion              = "x-ms-version"
	headerContentEncoding         = "Content-Encoding"
	headerContentLanguage         = "Content-Language"
	headerContentType             = "Content-Type"
	headerContentMD5              = "Content-MD5"
	headerIfModifiedSince         = "If-Modified-Since"
	headerIfMatch                 = "If-Match"
	headerIfNoneMatch             = "If-None-Match"
	headerIfUnmodifiedSince       = "If-Unmodified-Since"
	headerRange                   = "Range"
	headerDataServiceVersion      = "DataServiceVersion"
	headerMaxDataServiceVersion   = "MaxDataServiceVersion"
	headerContentTransferEncoding = "Content-Transfer-Encoding"
)

func (c *Client) addAuthorizationHeader(verb, url string, headers map[string]string, auth authentication) (map[string]string, error) {
	if !c.sasClient {
		authHeader, err := c.getSharedKey(verb, url, headers, auth)
		if err != nil {
			return nil, err
		}
		headers[headerAuthorization] = authHeader
	}
	return headers, nil
}

func (c *Client) getSharedKey(verb, url string, headers map[string]string, auth authentication) (string, error) {
	canRes, err := c.buildCanonicalizedResource(url, auth, false)
	if err != nil {
		return "", err
	}

	canString, err := buildCanonicalizedString(verb, headers, canRes, auth)
	if err != nil {
		return "", err
	}
	return c.createAuthorizationHeader(canString, auth), nil
}

func (c *Client) buildCanonicalizedResource(uri string, auth authentication, sas bool) (string, error) {
	errMsg := "buildCanonicalizedResource error: %s"
	u, err := url.Parse(uri)
	if err != nil {
		return "", fmt.Errorf(errMsg, err.Error())
	}

	cr := bytes.NewBufferString("")
	if c.accountName != StorageEmulatorAccountName || !sas {
		cr.WriteString("/")
		cr.WriteString(c.getCanonicalizedAccountName())
	}

	if len(u.Path) > 0 {
		// Any portion of the CanonicalizedResource string that is derived from
		// the resource's URI should be encoded exactly as it is in the URI.
		// -- https://msdn.microsoft.com/en-gb/library/azure/dd179428.aspx
		cr.WriteString(u.EscapedPath())
	}

	params, err := url.ParseQuery(u.RawQuery)
	if err != nil {
		return "", fmt.Errorf(errMsg, err.Error())
	}

	// See https://github.com/Azure/azure-storage-net/blob/master/Lib/Common/Core/Util/AuthenticationUtility.cs#L277
	if auth == sharedKey {
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

func (c *Client) getCanonicalizedAccountName() string {
	// since we may be trying to access a secondary storage account, we need to
	// remove the -secondary part of the storage name
	return strings.TrimSuffix(c.accountName, "-secondary")
}

func buildCanonicalizedString(verb string, headers map[string]string, canonicalizedResource string, auth authentication) (string, error) {
	contentLength := headers[headerContentLength]
	if contentLength == "0" {
		contentLength = ""
	}
	date := headers[headerDate]
	if v, ok := headers[headerXmsDate]; ok {
		if auth == sharedKey || auth == sharedKeyLite {
			date = ""
		} else {
			date = v
		}
	}
	var canString string
	switch auth {
	case sharedKey:
		canString = strings.Join([]string{
			verb,
			headers[headerContentEncoding],
			headers[headerContentLanguage],
			contentLength,
			headers[headerContentMD5],
			headers[headerContentType],
			date,
			headers[headerIfModifiedSince],
			headers[headerIfMatch],
			headers[headerIfNoneMatch],
			headers[headerIfUnmodifiedSince],
			headers[headerRange],
			buildCanonicalizedHeader(headers),
			canonicalizedResource,
		}, "\n")
	case sharedKeyForTable:
		canString = strings.Join([]string{
			verb,
			headers[headerContentMD5],
			headers[headerContentType],
			date,
			canonicalizedResource,
		}, "\n")
	case sharedKeyLite:
		canString = strings.Join([]string{
			verb,
			headers[headerContentMD5],
			headers[headerContentType],
			date,
			buildCanonicalizedHeader(headers),
			canonicalizedResource,
		}, "\n")
	case sharedKeyLiteForTable:
		canString = strings.Join([]string{
			date,
			canonicalizedResource,
		}, "\n")
	default:
		return "", fmt.Errorf("%s authentication is not supported yet", auth)
	}
	return canString, nil
}

func buildCanonicalizedHeader(headers map[string]string) string {
	cm := make(map[string]string)

	for k, v := range headers {
		headerName := strings.TrimSpace(strings.ToLower(k))
		if strings.HasPrefix(headerName, "x-ms-") {
			cm[headerName] = v
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

func (c *Client) createAuthorizationHeader(canonicalizedString string, auth authentication) string {
	signature := c.computeHmac256(canonicalizedString)
	var key string
	switch auth {
	case sharedKey, sharedKeyForTable:
		key = "SharedKey"
	case sharedKeyLite, sharedKeyLiteForTable:
		key = "SharedKeyLite"
	}
	return fmt.Sprintf("%s %s:%s", key, c.getCanonicalizedAccountName(), signature)
}
