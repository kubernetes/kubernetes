//go:build !spiffeid_charset_backcompat
// +build !spiffeid_charset_backcompat

package spiffeid

func isBackcompatTrustDomainChar(c uint8) bool {
	return false
}

func isBackcompatPathChar(c uint8) bool {
	return false
}
