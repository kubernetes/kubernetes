package spiffeid

import (
	"errors"
	"fmt"
	"net/url"
	"strings"
)

const (
	schemePrefix    = "spiffe://"
	schemePrefixLen = len(schemePrefix)
)

// FromPath returns a new SPIFFE ID in the given trust domain and with the
// given path. The supplied path must be a valid absolute path according to the
// SPIFFE specification.
// See https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE-ID.md#22-path
func FromPath(td TrustDomain, path string) (ID, error) {
	if err := ValidatePath(path); err != nil {
		return ID{}, err
	}
	return makeID(td, path)
}

// FromPathf returns a new SPIFFE ID from the formatted path in the given trust
// domain. The formatted path must be a valid absolute path according to the
// SPIFFE specification.
// See https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE-ID.md#22-path
func FromPathf(td TrustDomain, format string, args ...interface{}) (ID, error) {
	path, err := FormatPath(format, args...)
	if err != nil {
		return ID{}, err
	}
	return makeID(td, path)
}

// FromSegments returns a new SPIFFE ID in the given trust domain with joined
// path segments. The path segments must be valid according to the SPIFFE
// specification and must not contain path separators.
// See https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE-ID.md#22-path
func FromSegments(td TrustDomain, segments ...string) (ID, error) {
	path, err := JoinPathSegments(segments...)
	if err != nil {
		return ID{}, err
	}
	return makeID(td, path)
}

// FromString parses a SPIFFE ID from a string.
func FromString(id string) (ID, error) {
	switch {
	case id == "":
		return ID{}, errEmpty
	case !strings.HasPrefix(id, schemePrefix):
		return ID{}, errWrongScheme
	}

	pathidx := schemePrefixLen
	for ; pathidx < len(id); pathidx++ {
		c := id[pathidx]
		if c == '/' {
			break
		}
		if !isValidTrustDomainChar(c) {
			return ID{}, errBadTrustDomainChar
		}
	}

	if pathidx == schemePrefixLen {
		return ID{}, errMissingTrustDomain
	}

	if err := ValidatePath(id[pathidx:]); err != nil {
		return ID{}, err
	}

	return ID{
		id:      id,
		pathidx: pathidx,
	}, nil
}

// FromStringf parses a SPIFFE ID from a formatted string.
func FromStringf(format string, args ...interface{}) (ID, error) {
	return FromString(fmt.Sprintf(format, args...))
}

// FromURI parses a SPIFFE ID from a URI.
func FromURI(uri *url.URL) (ID, error) {
	return FromString(uri.String())
}

// ID is a SPIFFE ID
type ID struct {
	id string

	// pathidx tracks the index to the beginning of the path inside of id. This
	// is used when extracting the trust domain or path portions of the id.
	pathidx int
}

// TrustDomain returns the trust domain of the SPIFFE ID.
func (id ID) TrustDomain() TrustDomain {
	if id.IsZero() {
		return TrustDomain{}
	}
	return TrustDomain{name: id.id[schemePrefixLen:id.pathidx]}
}

// MemberOf returns true if the SPIFFE ID is a member of the given trust domain.
func (id ID) MemberOf(td TrustDomain) bool {
	return id.TrustDomain() == td
}

// Path returns the path of the SPIFFE ID inside the trust domain.
func (id ID) Path() string {
	return id.id[id.pathidx:]
}

// String returns the string representation of the SPIFFE ID, e.g.,
// "spiffe://example.org/foo/bar".
func (id ID) String() string {
	return id.id
}

// URL returns a URL for SPIFFE ID.
func (id ID) URL() *url.URL {
	if id.IsZero() {
		return &url.URL{}
	}

	return &url.URL{
		Scheme: "spiffe",
		Host:   id.TrustDomain().String(),
		Path:   id.Path(),
	}
}

// IsZero returns true if the SPIFFE ID is the zero value.
func (id ID) IsZero() bool {
	return id.id == ""
}

// AppendPath returns an ID with the appended path. It will fail if called on a
// zero value. The path to append must be a valid absolute path according to
// the SPIFFE specification.
// See https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE-ID.md#22-path
func (id ID) AppendPath(path string) (ID, error) {
	if id.IsZero() {
		return ID{}, errors.New("cannot append path on a zero ID value")
	}
	if err := ValidatePath(path); err != nil {
		return ID{}, err
	}
	id.id += path
	return id, nil
}

// AppendPathf returns an ID with the appended formatted path. It will fail if
// called on a zero value. The formatted path must be a valid absolute path
// according to the SPIFFE specification.
// See https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE-ID.md#22-path
func (id ID) AppendPathf(format string, args ...interface{}) (ID, error) {
	if id.IsZero() {
		return ID{}, errors.New("cannot append path on a zero ID value")
	}
	path, err := FormatPath(format, args...)
	if err != nil {
		return ID{}, err
	}
	id.id += path
	return id, nil
}

// AppendSegments returns an ID with the appended joined path segments.  It
// will fail if called on a zero value. The path segments must be valid
// according to the SPIFFE specification and must not contain path separators.
// See https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE-ID.md#22-path
func (id ID) AppendSegments(segments ...string) (ID, error) {
	if id.IsZero() {
		return ID{}, errors.New("cannot append path segments on a zero ID value")
	}
	path, err := JoinPathSegments(segments...)
	if err != nil {
		return ID{}, err
	}
	id.id += path
	return id, nil
}

// Replace path returns an ID with the given path in the same trust domain. It
// will fail if called on a zero value. The given path must be a valid absolute
// path according to the SPIFFE specification.
// See https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE-ID.md#22-path
func (id ID) ReplacePath(path string) (ID, error) {
	if id.IsZero() {
		return ID{}, errors.New("cannot replace path on a zero ID value")
	}
	return FromPath(id.TrustDomain(), path)
}

// ReplacePathf returns an ID with the formatted path in the same trust domain.
// It will fail if called on a zero value. The formatted path must be a valid
// absolute path according to the SPIFFE specification.
// See https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE-ID.md#22-path
func (id ID) ReplacePathf(format string, args ...interface{}) (ID, error) {
	if id.IsZero() {
		return ID{}, errors.New("cannot replace path on a zero ID value")
	}
	return FromPathf(id.TrustDomain(), format, args...)
}

// ReplaceSegments returns an ID with the joined path segments in the same
// trust domain. It will fail if called on a zero value. The path segments must
// be valid according to the SPIFFE specification and must not contain path
// separators.
// See https://github.com/spiffe/spiffe/blob/main/standards/SPIFFE-ID.md#22-path
func (id ID) ReplaceSegments(segments ...string) (ID, error) {
	if id.IsZero() {
		return ID{}, errors.New("cannot replace path segments on a zero ID value")
	}
	return FromSegments(id.TrustDomain(), segments...)
}

// MarshalText returns a text representation of the ID. If the ID is the zero
// value, nil is returned.
func (id ID) MarshalText() ([]byte, error) {
	if id.IsZero() {
		return nil, nil
	}
	return []byte(id.String()), nil
}

// UnmarshalText decodes a text representation of the ID. If the text is empty,
// the ID is set to the zero value.
func (id *ID) UnmarshalText(text []byte) error {
	if len(text) == 0 {
		*id = ID{}
		return nil
	}
	unmarshaled, err := FromString(string(text))
	if err != nil {
		return err
	}
	*id = unmarshaled
	return nil
}

func makeID(td TrustDomain, path string) (ID, error) {
	if td.IsZero() {
		return ID{}, errors.New("trust domain is empty")
	}
	return ID{
		id:      schemePrefix + td.name + path,
		pathidx: schemePrefixLen + len(td.name),
	}, nil
}
