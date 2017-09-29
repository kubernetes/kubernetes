package libtrust

import (
	"crypto"
	_ "crypto/sha256" // Registrer SHA224 and SHA256
	_ "crypto/sha512" // Registrer SHA384 and SHA512
	"fmt"
)

type signatureAlgorithm struct {
	algHeaderParam string
	hashID         crypto.Hash
}

func (h *signatureAlgorithm) HeaderParam() string {
	return h.algHeaderParam
}

func (h *signatureAlgorithm) HashID() crypto.Hash {
	return h.hashID
}

var (
	rs256 = &signatureAlgorithm{"RS256", crypto.SHA256}
	rs384 = &signatureAlgorithm{"RS384", crypto.SHA384}
	rs512 = &signatureAlgorithm{"RS512", crypto.SHA512}
	es256 = &signatureAlgorithm{"ES256", crypto.SHA256}
	es384 = &signatureAlgorithm{"ES384", crypto.SHA384}
	es512 = &signatureAlgorithm{"ES512", crypto.SHA512}
)

func rsaSignatureAlgorithmByName(alg string) (*signatureAlgorithm, error) {
	switch {
	case alg == "RS256":
		return rs256, nil
	case alg == "RS384":
		return rs384, nil
	case alg == "RS512":
		return rs512, nil
	default:
		return nil, fmt.Errorf("RSA Digital Signature Algorithm %q not supported", alg)
	}
}

func rsaPKCS1v15SignatureAlgorithmForHashID(hashID crypto.Hash) *signatureAlgorithm {
	switch {
	case hashID == crypto.SHA512:
		return rs512
	case hashID == crypto.SHA384:
		return rs384
	case hashID == crypto.SHA256:
		fallthrough
	default:
		return rs256
	}
}
