// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tls

import (
	"crypto"
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/rsa"
	"fmt"
)

// DigitallySigned gives information about a signature, including the algorithm used
// and the signature value.  Defined in RFC 5246 s4.7.
type DigitallySigned struct {
	Algorithm SignatureAndHashAlgorithm
	Signature []byte `tls:"minlen:0,maxlen:65535"`
}

func (d DigitallySigned) String() string {
	return fmt.Sprintf("Signature: HashAlgo=%v SignAlgo=%v Value=%x", d.Algorithm.Hash, d.Algorithm.Signature, d.Signature)
}

// SignatureAndHashAlgorithm gives information about the algorithms used for a
// signature.  Defined in RFC 5246 s7.4.1.4.1.
type SignatureAndHashAlgorithm struct {
	Hash      HashAlgorithm      `tls:"maxval:255"`
	Signature SignatureAlgorithm `tls:"maxval:255"`
}

// HashAlgorithm enum from RFC 5246 s7.4.1.4.1.
type HashAlgorithm Enum

// HashAlgorithm constants from RFC 5246 s7.4.1.4.1.
const (
	None   HashAlgorithm = 0
	MD5    HashAlgorithm = 1
	SHA1   HashAlgorithm = 2
	SHA224 HashAlgorithm = 3
	SHA256 HashAlgorithm = 4
	SHA384 HashAlgorithm = 5
	SHA512 HashAlgorithm = 6
)

func (h HashAlgorithm) String() string {
	switch h {
	case None:
		return "None"
	case MD5:
		return "MD5"
	case SHA1:
		return "SHA1"
	case SHA224:
		return "SHA224"
	case SHA256:
		return "SHA256"
	case SHA384:
		return "SHA384"
	case SHA512:
		return "SHA512"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", h)
	}
}

// SignatureAlgorithm enum from RFC 5246 s7.4.1.4.1.
type SignatureAlgorithm Enum

// SignatureAlgorithm constants from RFC 5246 s7.4.1.4.1.
const (
	Anonymous SignatureAlgorithm = 0
	RSA       SignatureAlgorithm = 1
	DSA       SignatureAlgorithm = 2
	ECDSA     SignatureAlgorithm = 3
)

func (s SignatureAlgorithm) String() string {
	switch s {
	case Anonymous:
		return "Anonymous"
	case RSA:
		return "RSA"
	case DSA:
		return "DSA"
	case ECDSA:
		return "ECDSA"
	default:
		return fmt.Sprintf("UNKNOWN(%d)", s)
	}
}

// SignatureAlgorithmFromPubKey returns the algorithm used for this public key.
// ECDSA, RSA, and DSA keys are supported. Other key types will return Anonymous.
func SignatureAlgorithmFromPubKey(k crypto.PublicKey) SignatureAlgorithm {
	switch k.(type) {
	case *ecdsa.PublicKey:
		return ECDSA
	case *rsa.PublicKey:
		return RSA
	case *dsa.PublicKey:
		return DSA
	default:
		return Anonymous
	}
}
