// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
package host

import (
	"github.com/microsoft/wmi/pkg/base/credential"
)

type WmiHost struct {
	HostName   string
	credential *credential.WmiCredential
}

func NewWmiLocalHost() *WmiHost {
	return NewWmiHost("localhost")
}

// NewWmiHost
func NewWmiHost(hostname string) *WmiHost {
	return NewWmiHostWithCredential(hostname, "", "", "")
}

// NewWmiHostWithCredential
func NewWmiHostWithCredential(hostname, username, password, domain string) *WmiHost {
	return &WmiHost{HostName: hostname, credential: credential.NewWmiCredential(username, password, domain)}
}

// GetCredential
func (host *WmiHost) GetCredential() *credential.WmiCredential {
	return host.credential
}
