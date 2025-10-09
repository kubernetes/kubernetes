package spiffeid

import (
	"net/url"
	"strings"
)

// TrustDomain represents the trust domain portion of a SPIFFE ID (e.g.
// example.org).
type TrustDomain struct {
	name string
}

// TrustDomainFromString returns a new TrustDomain from a string. The string
// can either be a trust domain name (e.g. example.org), or a valid SPIFFE ID
// URI (e.g. spiffe://example.org), otherwise an error is returned.
// See https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE-ID.md#21-trust-domain.
func TrustDomainFromString(idOrName string) (TrustDomain, error) {
	switch {
	case idOrName == "":
		return TrustDomain{}, errMissingTrustDomain
	case strings.Contains(idOrName, ":/"):
		// The ID looks like it has something like a scheme separator, let's
		// try to parse as an ID. We use :/ instead of :// since the
		// diagnostics are better for a bad input like spiffe:/trustdomain.
		id, err := FromString(idOrName)
		if err != nil {
			return TrustDomain{}, err
		}
		return id.TrustDomain(), nil
	default:
		for i := 0; i < len(idOrName); i++ {
			if !isValidTrustDomainChar(idOrName[i]) {
				return TrustDomain{}, errBadTrustDomainChar
			}
		}
		return TrustDomain{name: idOrName}, nil
	}
}

// TrustDomainFromURI returns a new TrustDomain from a URI. The URI must be a
// valid SPIFFE ID (see FromURI) or an error is returned. The trust domain is
// extracted from the host field.
func TrustDomainFromURI(uri *url.URL) (TrustDomain, error) {
	id, err := FromURI(uri)
	if err != nil {
		return TrustDomain{}, err
	}

	return id.TrustDomain(), nil
}

// Name returns the trust domain name as a string, e.g. example.org.
func (td TrustDomain) Name() string {
	return td.name
}

// String returns the trust domain name as a string, e.g. example.org.
func (td TrustDomain) String() string {
	return td.name
}

// ID returns the SPIFFE ID of the trust domain.
func (td TrustDomain) ID() ID {
	if id, err := makeID(td, ""); err == nil {
		return id
	}
	return ID{}
}

// IDString returns a string representation of the the SPIFFE ID of the trust
// domain, e.g. "spiffe://example.org".
func (td TrustDomain) IDString() string {
	return td.ID().String()
}

// IsZero returns true if the trust domain is the zero value.
func (td TrustDomain) IsZero() bool {
	return td.name == ""
}

// Compare returns an integer comparing the trust domain to another
// lexicographically. The result will be 0 if td==other, -1 if td < other, and
// +1 if td > other.
func (td TrustDomain) Compare(other TrustDomain) int {
	return strings.Compare(td.name, other.name)
}

// MarshalText returns a text representation of the trust domain. If the trust
// domain is the zero value, nil is returned.
func (td TrustDomain) MarshalText() ([]byte, error) {
	if td.IsZero() {
		return nil, nil
	}
	return []byte(td.String()), nil
}

// UnmarshalText decodes a text representation of the trust domain. If the text
// is empty, the trust domain is set to the zero value.
func (td *TrustDomain) UnmarshalText(text []byte) error {
	if len(text) == 0 {
		*td = TrustDomain{}
		return nil
	}

	unmarshaled, err := TrustDomainFromString(string(text))
	if err != nil {
		return err
	}
	*td = unmarshaled
	return nil
}

func isValidTrustDomainChar(c uint8) bool {
	switch {
	case c >= 'a' && c <= 'z':
		return true
	case c >= '0' && c <= '9':
		return true
	case c == '-', c == '.', c == '_':
		return true
	case isBackcompatTrustDomainChar(c):
		return true
	default:
		return false
	}
}
