/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package bootstrap

import (
	"fmt"
	"strings"

	jose "github.com/square/go-jose"
)

// computeDetachedSig takes content and token details and computes a detached
// JWS signature.  This is described in Appendix F of RFC 7515.  Basically, this
// is a regular JWS with the content part of the signature elided.
func computeDetachedSig(content, tokenID, tokenSecret string) (string, error) {
	jwk := &jose.JsonWebKey{
		Key:   []byte(tokenSecret),
		KeyID: tokenID,
	}

	signer, err := jose.NewSigner(jose.HS256, jwk)
	if err != nil {
		return "", nil
	}

	jws, err := signer.Sign([]byte(content))
	if err != nil {
		return "", nil
	}

	fullSig, err := jws.CompactSerialize()
	if err != nil {
		return "", nil
	}
	return stripContent(fullSig)
}

// stripContent will remove the content part of a compact JWS
//
// The `go-jose` library doesn't support generating signatures with "detatched"
// content. To make up for this we take the full compact signature, break it
// apart and put it back together without the content section.
func stripContent(fullSig string) (string, error) {
	parts := strings.Split(fullSig, ".")
	if len(parts) != 3 {
		return "", fmt.Errorf("Compact JWS format must have three parts")
	}

	return parts[0] + ".." + parts[2], nil
}
