// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
package credential

type WmiCredential struct {
	UserName string
	Password string
	Domain   string
}

// NewWmiCredential
func NewWmiCredential(username, password, domain string) *WmiCredential {
	return &WmiCredential{UserName: username, Password: password, Domain: domain}
}
