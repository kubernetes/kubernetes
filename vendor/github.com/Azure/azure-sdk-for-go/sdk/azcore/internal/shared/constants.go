//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package shared

const (
	ContentTypeAppJSON = "application/json"
	ContentTypeAppXML  = "application/xml"
)

const (
	HeaderAuthorization          = "Authorization"
	HeaderAuxiliaryAuthorization = "x-ms-authorization-auxiliary"
	HeaderAzureAsync             = "Azure-AsyncOperation"
	HeaderContentLength          = "Content-Length"
	HeaderContentType            = "Content-Type"
	HeaderLocation               = "Location"
	HeaderOperationLocation      = "Operation-Location"
	HeaderRetryAfter             = "Retry-After"
	HeaderUserAgent              = "User-Agent"
)

const BearerTokenPrefix = "Bearer "

const (
	// Module is the name of the calling module used in telemetry data.
	Module = "azcore"

	// Version is the semantic version (see http://semver.org) of this module.
	Version = "v1.0.0"
)
