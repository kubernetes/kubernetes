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

// OverrideHeaders defines overridable response heaedrs in
// a request using a SAS URI.
// See https://docs.microsoft.com/en-us/rest/api/storageservices/constructing-a-service-sas
type OverrideHeaders struct {
	CacheControl       string
	ContentDisposition string
	ContentEncoding    string
	ContentLanguage    string
	ContentType        string
}

// BlobSASOptions are options to construct a blob SAS
// URI.
// See https://docs.microsoft.com/en-us/rest/api/storageservices/constructing-a-service-sas
type BlobSASOptions struct {
	BlobServiceSASPermissions
	OverrideHeaders
	SASOptions
}

// BlobServiceSASPermissions includes the available permissions for
// blob service SAS URI.
type BlobServiceSASPermissions struct {
	Read   bool
	Add    bool
	Create bool
	Write  bool
	Delete bool
}

func (p BlobServiceSASPermissions) buildString() string {
	permissions := ""
	if p.Read {
		permissions += "r"
	}
	if p.Add {
		permissions += "a"
	}
	if p.Create {
		permissions += "c"
	}
	if p.Write {
		permissions += "w"
	}
	if p.Delete {
		permissions += "d"
	}
	return permissions
}

// GetSASURI creates an URL to the blob which contains the Shared
// Access Signature with the specified options.
//
// See https://docs.microsoft.com/en-us/rest/api/storageservices/constructing-a-service-sas
func (b *Blob) GetSASURI(options BlobSASOptions) (string, error) {
	uri := b.GetURL()
	signedResource := "b"
	canonicalizedResource, err := b.Container.bsc.client.buildCanonicalizedResource(uri, b.Container.bsc.auth, true)
	if err != nil {
		return "", err
	}

	permissions := options.BlobServiceSASPermissions.buildString()
	return b.Container.bsc.client.blobAndFileSASURI(options.SASOptions, uri, permissions, canonicalizedResource, signedResource, options.OverrideHeaders)
}

func (c *Client) blobAndFileSASURI(options SASOptions, uri, permissions, canonicalizedResource, signedResource string, headers OverrideHeaders) (string, error) {
	start := ""
	if options.Start != (time.Time{}) {
		start = options.Start.UTC().Format(time.RFC3339)
	}

	expiry := options.Expiry.UTC().Format(time.RFC3339)

	// We need to replace + with %2b first to avoid being treated as a space (which is correct for query strings, but not the path component).
	canonicalizedResource = strings.Replace(canonicalizedResource, "+", "%2b", -1)
	canonicalizedResource, err := url.QueryUnescape(canonicalizedResource)
	if err != nil {
		return "", err
	}

	protocols := ""
	if options.UseHTTPS {
		protocols = "https"
	}
	stringToSign, err := blobSASStringToSign(permissions, start, expiry, canonicalizedResource, options.Identifier, options.IP, protocols, c.apiVersion, signedResource, "", headers)
	if err != nil {
		return "", err
	}

	sig := c.computeHmac256(stringToSign)
	sasParams := url.Values{
		"sv":  {c.apiVersion},
		"se":  {expiry},
		"sr":  {signedResource},
		"sp":  {permissions},
		"sig": {sig},
	}

	if start != "" {
		sasParams.Add("st", start)
	}

	if c.apiVersion >= "2015-04-05" {
		if protocols != "" {
			sasParams.Add("spr", protocols)
		}
		if options.IP != "" {
			sasParams.Add("sip", options.IP)
		}
	}

	// Add override response hedaers
	addQueryParameter(sasParams, "rscc", headers.CacheControl)
	addQueryParameter(sasParams, "rscd", headers.ContentDisposition)
	addQueryParameter(sasParams, "rsce", headers.ContentEncoding)
	addQueryParameter(sasParams, "rscl", headers.ContentLanguage)
	addQueryParameter(sasParams, "rsct", headers.ContentType)

	sasURL, err := url.Parse(uri)
	if err != nil {
		return "", err
	}
	sasURL.RawQuery = sasParams.Encode()
	return sasURL.String(), nil
}

func blobSASStringToSign(signedPermissions, signedStart, signedExpiry, canonicalizedResource, signedIdentifier, signedIP, protocols, signedVersion, signedResource, signedSnapshotTime string, headers OverrideHeaders) (string, error) {
	rscc := headers.CacheControl
	rscd := headers.ContentDisposition
	rsce := headers.ContentEncoding
	rscl := headers.ContentLanguage
	rsct := headers.ContentType

	if signedVersion >= "2015-02-21" {
		canonicalizedResource = "/blob" + canonicalizedResource
	}

	// https://docs.microsoft.com/en-us/rest/api/storageservices/constructing-a-service-sas
	if signedVersion >= "2018-11-09" {
		return fmt.Sprintf("%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s", signedPermissions, signedStart, signedExpiry, canonicalizedResource, signedIdentifier, signedIP, protocols, signedVersion, signedResource, signedSnapshotTime, rscc, rscd, rsce, rscl, rsct), nil
	}

	// https://msdn.microsoft.com/en-us/library/azure/dn140255.aspx#Anchor_12
	if signedVersion >= "2015-04-05" {
		return fmt.Sprintf("%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s", signedPermissions, signedStart, signedExpiry, canonicalizedResource, signedIdentifier, signedIP, protocols, signedVersion, rscc, rscd, rsce, rscl, rsct), nil
	}

	// reference: http://msdn.microsoft.com/en-us/library/azure/dn140255.aspx
	if signedVersion >= "2013-08-15" {
		return fmt.Sprintf("%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s", signedPermissions, signedStart, signedExpiry, canonicalizedResource, signedIdentifier, signedVersion, rscc, rscd, rsce, rscl, rsct), nil
	}

	return "", errors.New("storage: not implemented SAS for versions earlier than 2013-08-15")
}
