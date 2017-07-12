package storage

import (
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"
)

// GetSASURIWithSignedIPAndProtocol creates an URL to the specified blob which contains the Shared
// Access Signature with specified permissions and expiration time. Also includes signedIPRange and allowed protocols.
// If old API version is used but no signedIP is passed (ie empty string) then this should still work.
// We only populate the signedIP when it non-empty.
//
// See https://msdn.microsoft.com/en-us/library/azure/ee395415.aspx
func (b *Blob) GetSASURIWithSignedIPAndProtocol(expiry time.Time, permissions string, signedIPRange string, HTTPSOnly bool) (string, error) {
	var (
		signedPermissions = permissions
		blobURL           = b.GetURL()
	)
	canonicalizedResource, err := b.Container.bsc.client.buildCanonicalizedResource(blobURL, b.Container.bsc.auth)
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

	signedExpiry := expiry.UTC().Format(time.RFC3339)

	//If blob name is missing, resource is a container
	signedResource := "c"
	if len(b.Name) > 0 {
		signedResource = "b"
	}

	protocols := "https,http"
	if HTTPSOnly {
		protocols = "https"
	}
	stringToSign, err := blobSASStringToSign(b.Container.bsc.client.apiVersion, canonicalizedResource, signedExpiry, signedPermissions, signedIPRange, protocols)
	if err != nil {
		return "", err
	}

	sig := b.Container.bsc.client.computeHmac256(stringToSign)
	sasParams := url.Values{
		"sv":  {b.Container.bsc.client.apiVersion},
		"se":  {signedExpiry},
		"sr":  {signedResource},
		"sp":  {signedPermissions},
		"sig": {sig},
	}

	if b.Container.bsc.client.apiVersion >= "2015-04-05" {
		sasParams.Add("spr", protocols)
		if signedIPRange != "" {
			sasParams.Add("sip", signedIPRange)
		}
	}

	sasURL, err := url.Parse(blobURL)
	if err != nil {
		return "", err
	}
	sasURL.RawQuery = sasParams.Encode()
	return sasURL.String(), nil
}

// GetSASURI creates an URL to the specified blob which contains the Shared
// Access Signature with specified permissions and expiration time.
//
// See https://msdn.microsoft.com/en-us/library/azure/ee395415.aspx
func (b *Blob) GetSASURI(expiry time.Time, permissions string) (string, error) {
	return b.GetSASURIWithSignedIPAndProtocol(expiry, permissions, "", false)
}

func blobSASStringToSign(signedVersion, canonicalizedResource, signedExpiry, signedPermissions string, signedIP string, protocols string) (string, error) {
	var signedStart, signedIdentifier, rscc, rscd, rsce, rscl, rsct string

	if signedVersion >= "2015-02-21" {
		canonicalizedResource = "/blob" + canonicalizedResource
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
