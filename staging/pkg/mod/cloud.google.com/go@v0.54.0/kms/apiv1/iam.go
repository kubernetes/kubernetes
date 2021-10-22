// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kms

import (
	"cloud.google.com/go/iam"
	kmspb "google.golang.org/genproto/googleapis/cloud/kms/v1"
)

// KeyRingIAM returns a handle to inspect and change permissions of a KeyRing.
//
// Deprecated: Please use ResourceIAM and provide the KeyRing.Name as input.
func (c *KeyManagementClient) KeyRingIAM(keyRing *kmspb.KeyRing) *iam.Handle {
	return iam.InternalNewHandle(c.Connection(), keyRing.Name)
}

// CryptoKeyIAM returns a handle to inspect and change permissions of a CryptoKey.
//
// Deprecated: Please use ResourceIAM and provide the CryptoKey.Name as input.
func (c *KeyManagementClient) CryptoKeyIAM(cryptoKey *kmspb.CryptoKey) *iam.Handle {
	return iam.InternalNewHandle(c.Connection(), cryptoKey.Name)
}

// ResourceIAM returns a handle to inspect and change permissions of the resource
// indicated by the given resource path.
func (c *KeyManagementClient) ResourceIAM(resourcePath string) *iam.Handle {
	return iam.InternalNewHandle(c.Connection(), resourcePath)
}
