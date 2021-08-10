package storage

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"strings"
	"time"
)

// AccessPolicyDetailsXML has specifics about an access policy
// annotated with XML details.
type AccessPolicyDetailsXML struct {
	StartTime  time.Time `xml:"Start"`
	ExpiryTime time.Time `xml:"Expiry"`
	Permission string    `xml:"Permission"`
}

// SignedIdentifier is a wrapper for a specific policy
type SignedIdentifier struct {
	ID           string                 `xml:"Id"`
	AccessPolicy AccessPolicyDetailsXML `xml:"AccessPolicy"`
}

// SignedIdentifiers part of the response from GetPermissions call.
type SignedIdentifiers struct {
	SignedIdentifiers []SignedIdentifier `xml:"SignedIdentifier"`
}

// AccessPolicy is the response type from the GetPermissions call.
type AccessPolicy struct {
	SignedIdentifiersList SignedIdentifiers `xml:"SignedIdentifiers"`
}

// convertAccessPolicyToXMLStructs converts between AccessPolicyDetails which is a struct better for API usage to the
// AccessPolicy struct which will get converted to XML.
func convertAccessPolicyToXMLStructs(id string, startTime time.Time, expiryTime time.Time, permissions string) SignedIdentifier {
	return SignedIdentifier{
		ID: id,
		AccessPolicy: AccessPolicyDetailsXML{
			StartTime:  startTime.UTC().Round(time.Second),
			ExpiryTime: expiryTime.UTC().Round(time.Second),
			Permission: permissions,
		},
	}
}

func updatePermissions(permissions, permission string) bool {
	return strings.Contains(permissions, permission)
}
