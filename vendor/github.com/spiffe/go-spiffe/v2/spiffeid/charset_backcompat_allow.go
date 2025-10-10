//go:build spiffeid_charset_backcompat
// +build spiffeid_charset_backcompat

package spiffeid

func isBackcompatTrustDomainChar(c uint8) bool {
	if isSubDelim(c) {
		return true
	}
	switch c {
	// unreserved
	case '~':
		return true
	default:
		return false
	}
}

func isBackcompatPathChar(c uint8) bool {
	if isSubDelim(c) {
		return true
	}
	switch c {
	// unreserved
	case '~':
		return true
	// gen-delims
	case ':', '[', ']', '@':
		return true
	default:
		return false
	}
}

func isSubDelim(c uint8) bool {
	switch c {
	case '!', '$', '&', '\'', '(', ')', '*', '+', ',', ';', '=':
		return true
	default:
		return false
	}
}
