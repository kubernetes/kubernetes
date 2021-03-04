package dns

import (
	"crypto/sha256"
	"crypto/sha512"
	"crypto/x509"
	"encoding/hex"
	"errors"
)

// CertificateToDANE converts a certificate to a hex string as used in the TLSA or SMIMEA records.
func CertificateToDANE(selector, matchingType uint8, cert *x509.Certificate) (string, error) {
	switch matchingType {
	case 0:
		switch selector {
		case 0:
			return hex.EncodeToString(cert.Raw), nil
		case 1:
			return hex.EncodeToString(cert.RawSubjectPublicKeyInfo), nil
		}
	case 1:
		h := sha256.New()
		switch selector {
		case 0:
			h.Write(cert.Raw)
			return hex.EncodeToString(h.Sum(nil)), nil
		case 1:
			h.Write(cert.RawSubjectPublicKeyInfo)
			return hex.EncodeToString(h.Sum(nil)), nil
		}
	case 2:
		h := sha512.New()
		switch selector {
		case 0:
			h.Write(cert.Raw)
			return hex.EncodeToString(h.Sum(nil)), nil
		case 1:
			h.Write(cert.RawSubjectPublicKeyInfo)
			return hex.EncodeToString(h.Sum(nil)), nil
		}
	}
	return "", errors.New("dns: bad MatchingType or Selector")
}
