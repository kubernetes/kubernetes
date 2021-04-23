package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"
)

// QueueSASOptions are options to construct a blob SAS
// URI.
// See https://docs.microsoft.com/en-us/rest/api/storageservices/constructing-a-service-sas
type QueueSASOptions struct {
	QueueSASPermissions
	SASOptions
}

// QueueSASPermissions includes the available permissions for
// a queue SAS URI.
type QueueSASPermissions struct {
	Read    bool
	Add     bool
	Update  bool
	Process bool
}

func (q QueueSASPermissions) buildString() string {
	permissions := ""

	if q.Read {
		permissions += "r"
	}
	if q.Add {
		permissions += "a"
	}
	if q.Update {
		permissions += "u"
	}
	if q.Process {
		permissions += "p"
	}
	return permissions
}

// GetSASURI creates an URL to the specified queue which contains the Shared
// Access Signature with specified permissions and expiration time.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/constructing-a-service-sas
func (q *Queue) GetSASURI(options QueueSASOptions) (string, error) {
	canonicalizedResource, err := q.qsc.client.buildCanonicalizedResource(q.buildPath(), q.qsc.auth, true)
	if err != nil {
		return "", err
	}

	// "The canonicalizedresouce portion of the string is a canonical path to the signed resource.
	// It must include the service name (blob, table, queue or file) for version 2015-02-21 or
	// later, the storage account name, and the resource name, and must be URL-decoded.
	// -- https://msdn.microsoft.com/en-us/library/azure/dn140255.aspx
	// We need to replace + with %2b first to avoid being treated as a space (which is correct for query strings, but not the path component).
	canonicalizedResource = strings.Replace(canonicalizedResource, "+", "%2b", -1)
	canonicalizedResource, err = url.QueryUnescape(canonicalizedResource)
	if err != nil {
		return "", err
	}

	signedStart := ""
	if options.Start != (time.Time{}) {
		signedStart = options.Start.UTC().Format(time.RFC3339)
	}
	signedExpiry := options.Expiry.UTC().Format(time.RFC3339)

	protocols := "https,http"
	if options.UseHTTPS {
		protocols = "https"
	}

	permissions := options.QueueSASPermissions.buildString()
	stringToSign, err := queueSASStringToSign(q.qsc.client.apiVersion, canonicalizedResource, signedStart, signedExpiry, options.IP, permissions, protocols, options.Identifier)
	if err != nil {
		return "", err
	}

	sig := q.qsc.client.computeHmac256(stringToSign)
	sasParams := url.Values{
		"sv":  {q.qsc.client.apiVersion},
		"se":  {signedExpiry},
		"sp":  {permissions},
		"sig": {sig},
	}

	if q.qsc.client.apiVersion >= "2015-04-05" {
		sasParams.Add("spr", protocols)
		addQueryParameter(sasParams, "sip", options.IP)
	}

	uri := q.qsc.client.getEndpoint(queueServiceName, q.buildPath(), nil)
	sasURL, err := url.Parse(uri)
	if err != nil {
		return "", err
	}
	sasURL.RawQuery = sasParams.Encode()
	return sasURL.String(), nil
}

func queueSASStringToSign(signedVersion, canonicalizedResource, signedStart, signedExpiry, signedIP, signedPermissions, protocols, signedIdentifier string) (string, error) {

	if signedVersion >= "2015-02-21" {
		canonicalizedResource = "/queue" + canonicalizedResource
	}

	// https://msdn.microsoft.com/en-us/library/azure/dn140255.aspx#Anchor_12
	if signedVersion >= "2015-04-05" {
		return fmt.Sprintf("%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s",
			signedPermissions,
			signedStart,
			signedExpiry,
			canonicalizedResource,
			signedIdentifier,
			signedIP,
			protocols,
			signedVersion), nil

	}

	// reference: http://msdn.microsoft.com/en-us/library/azure/dn140255.aspx
	if signedVersion >= "2013-08-15" {
		return fmt.Sprintf("%s\n%s\n%s\n%s\n%s\n%s", signedPermissions, signedStart, signedExpiry, canonicalizedResource, signedIdentifier, signedVersion), nil
	}

	return "", errors.New("storage: not implemented SAS for versions earlier than 2013-08-15")
}
