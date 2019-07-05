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
