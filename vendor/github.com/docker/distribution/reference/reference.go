// Package reference provides a general type to represent any way of referencing images within the registry.
// Its main purpose is to abstract tags and digests (content-addressable hash).
//
// Grammar
//
//	reference                       := name [ ":" tag ] [ "@" digest ]
//	name                            := [domain '/'] path-component ['/' path-component]*
//	domain                          := domain-component ['.' domain-component]* [':' port-number]
//	domain-component                := /([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])/
//	port-number                     := /[0-9]+/
//	path-component                  := alpha-numeric [separator alpha-numeric]*
//	alpha-numeric                   := /[a-z0-9]+/
//	separator                       := /[_.]|__|[-]*/
//
//	tag                             := /[\w][\w.-]{0,127}/
//
//	digest                          := digest-algorithm ":" digest-hex
//	digest-algorithm                := digest-algorithm-component [ digest-algorithm-separator digest-algorithm-component ]*
//	digest-algorithm-separator      := /[+.-_]/
//	digest-algorithm-component      := /[A-Za-z][A-Za-z0-9]*/
//	digest-hex                      := /[0-9a-fA-F]{32,}/ ; At least 128 bit digest value
//
//	identifier                      := /[a-f0-9]{64}/
//	short-identifier                := /[a-f0-9]{6,64}/
package reference

import (
	"errors"
	"fmt"
	"strings"

	"github.com/opencontainers/go-digest"
)

const (
	// NameTotalLengthMax is the maximum total number of characters in a repository name.
	NameTotalLengthMax = 255
)

var (
	// ErrReferenceInvalidFormat represents an error while trying to parse a string as a reference.
	ErrReferenceInvalidFormat = errors.New("invalid reference format")

	// ErrTagInvalidFormat represents an error while trying to parse a string as a tag.
	ErrTagInvalidFormat = errors.New("invalid tag format")

	// ErrDigestInvalidFormat represents an error while trying to parse a string as a tag.
	ErrDigestInvalidFormat = errors.New("invalid digest format")

	// ErrNameContainsUppercase is returned for invalid repository names that contain uppercase characters.
	ErrNameContainsUppercase = errors.New("repository name must be lowercase")

	// ErrNameEmpty is returned for empty, invalid repository names.
	ErrNameEmpty = errors.New("repository name must have at least one component")

	// ErrNameTooLong is returned when a repository name is longer than NameTotalLengthMax.
	ErrNameTooLong = fmt.Errorf("repository name must not be more than %v characters", NameTotalLengthMax)

	// ErrNameNotCanonical is returned when a name is not canonical.
	ErrNameNotCanonical = errors.New("repository name must be canonical")
)

// Reference is an opaque object reference identifier that may include
// modifiers such as a hostname, name, tag, and digest.
type Reference interface {
	// String returns the full reference
	String() string
}

// Field provides a wrapper type for resolving correct reference types when
// working with encoding.
type Field struct {
	reference Reference
}

// AsField wraps a reference in a Field for encoding.
func AsField(reference Reference) Field {
	return Field{reference}
}

// Reference unwraps the reference type from the field to
// return the Reference object. This object should be
// of the appropriate type to further check for different
// reference types.
func (f Field) Reference() Reference {
	return f.reference
}

// MarshalText serializes the field to byte text which
// is the string of the reference.
func (f Field) MarshalText() (p []byte, err error) {
	return []byte(f.reference.String()), nil
}

// UnmarshalText parses text bytes by invoking the
// reference parser to ensure the appropriately
// typed reference object is wrapped by field.
func (f *Field) UnmarshalText(p []byte) error {
	r, err := Parse(string(p))
	if err != nil {
		return err
	}

	f.reference = r
	return nil
}

// Named is an object with a full name
type Named interface {
	Reference
	Name() string
}

// Tagged is an object which has a tag
type Tagged interface {
	Reference
	Tag() string
}

// NamedTagged is an object including a name and tag.
type NamedTagged interface {
	Named
	Tag() string
}

// Digested is an object which has a digest
// in which it can be referenced by
type Digested interface {
	Reference
	Digest() digest.Digest
}

// Canonical reference is an object with a fully unique
// name including a name with domain and digest
type Canonical interface {
	Named
	Digest() digest.Digest
}

// namedRepository is a reference to a repository with a name.
// A namedRepository has both domain and path components.
type namedRepository interface {
	Named
	Domain() string
	Path() string
}

// Domain returns the domain part of the Named reference
func Domain(named Named) string {
	if r, ok := named.(namedRepository); ok {
		return r.Domain()
	}
	domain, _ := splitDomain(named.Name())
	return domain
}

// Path returns the name without the domain part of the Named reference
func Path(named Named) (name string) {
	if r, ok := named.(namedRepository); ok {
		return r.Path()
	}
	_, path := splitDomain(named.Name())
	return path
}

func splitDomain(name string) (string, string) {
	match := anchoredNameRegexp.FindStringSubmatch(name)
	if len(match) != 3 {
		return "", name
	}
	return match[1], match[2]
}

// SplitHostname splits a named reference into a
// hostname and name string. If no valid hostname is
// found, the hostname is empty and the full value
// is returned as name
// DEPRECATED: Use Domain or Path
func SplitHostname(named Named) (string, string) {
	if r, ok := named.(namedRepository); ok {
		return r.Domain(), r.Path()
	}
	return splitDomain(named.Name())
}

// Parse parses s and returns a syntactically valid Reference.
// If an error was encountered it is returned, along with a nil Reference.
// NOTE: Parse will not handle short digests.
func Parse(s string) (Reference, error) {
	matches := ReferenceRegexp.FindStringSubmatch(s)
	if matches == nil {
		if s == "" {
			return nil, ErrNameEmpty
		}
		if ReferenceRegexp.FindStringSubmatch(strings.ToLower(s)) != nil {
			return nil, ErrNameContainsUppercase
		}
		return nil, ErrReferenceInvalidFormat
	}

	if len(matches[1]) > NameTotalLengthMax {
		return nil, ErrNameTooLong
	}

	var repo repository

	nameMatch := anchoredNameRegexp.FindStringSubmatch(matches[1])
	if len(nameMatch) == 3 {
		repo.domain = nameMatch[1]
		repo.path = nameMatch[2]
	} else {
		repo.domain = ""
		repo.path = matches[1]
	}

	ref := reference{
		namedRepository: repo,
		tag:             matches[2],
	}
	if matches[3] != "" {
		var err error
		ref.digest, err = digest.Parse(matches[3])
		if err != nil {
			return nil, err
		}
	}

	r := getBestReferenceType(ref)
	if r == nil {
		return nil, ErrNameEmpty
	}

	return r, nil
}

// ParseNamed parses s and returns a syntactically valid reference implementing
// the Named interface. The reference must have a name and be in the canonical
// form, otherwise an error is returned.
// If an error was encountered it is returned, along with a nil Reference.
// NOTE: ParseNamed will not handle short digests.
func ParseNamed(s string) (Named, error) {
	named, err := ParseNormalizedNamed(s)
	if err != nil {
		return nil, err
	}
	if named.String() != s {
		return nil, ErrNameNotCanonical
	}
	return named, nil
}

// WithName returns a named object representing the given string. If the input
// is invalid ErrReferenceInvalidFormat will be returned.
func WithName(name string) (Named, error) {
	if len(name) > NameTotalLengthMax {
		return nil, ErrNameTooLong
	}

	match := anchoredNameRegexp.FindStringSubmatch(name)
	if match == nil || len(match) != 3 {
		return nil, ErrReferenceInvalidFormat
	}
	return repository{
		domain: match[1],
		path:   match[2],
	}, nil
}

// WithTag combines the name from "name" and the tag from "tag" to form a
// reference incorporating both the name and the tag.
func WithTag(name Named, tag string) (NamedTagged, error) {
	if !anchoredTagRegexp.MatchString(tag) {
		return nil, ErrTagInvalidFormat
	}
	var repo repository
	if r, ok := name.(namedRepository); ok {
		repo.domain = r.Domain()
		repo.path = r.Path()
	} else {
		repo.path = name.Name()
	}
	if canonical, ok := name.(Canonical); ok {
		return reference{
			namedRepository: repo,
			tag:             tag,
			digest:          canonical.Digest(),
		}, nil
	}
	return taggedReference{
		namedRepository: repo,
		tag:             tag,
	}, nil
}

// WithDigest combines the name from "name" and the digest from "digest" to form
// a reference incorporating both the name and the digest.
func WithDigest(name Named, digest digest.Digest) (Canonical, error) {
	if !anchoredDigestRegexp.MatchString(digest.String()) {
		return nil, ErrDigestInvalidFormat
	}
	var repo repository
	if r, ok := name.(namedRepository); ok {
		repo.domain = r.Domain()
		repo.path = r.Path()
	} else {
		repo.path = name.Name()
	}
	if tagged, ok := name.(Tagged); ok {
		return reference{
			namedRepository: repo,
			tag:             tagged.Tag(),
			digest:          digest,
		}, nil
	}
	return canonicalReference{
		namedRepository: repo,
		digest:          digest,
	}, nil
}

// TrimNamed removes any tag or digest from the named reference.
func TrimNamed(ref Named) Named {
	domain, path := SplitHostname(ref)
	return repository{
		domain: domain,
		path:   path,
	}
}

func getBestReferenceType(ref reference) Reference {
	if ref.Name() == "" {
		// Allow digest only references
		if ref.digest != "" {
			return digestReference(ref.digest)
		}
		return nil
	}
	if ref.tag == "" {
		if ref.digest != "" {
			return canonicalReference{
				namedRepository: ref.namedRepository,
				digest:          ref.digest,
			}
		}
		return ref.namedRepository
	}
	if ref.digest == "" {
		return taggedReference{
			namedRepository: ref.namedRepository,
			tag:             ref.tag,
		}
	}

	return ref
}

type reference struct {
	namedRepository
	tag    string
	digest digest.Digest
}

func (r reference) String() string {
	return r.Name() + ":" + r.tag + "@" + r.digest.String()
}

func (r reference) Tag() string {
	return r.tag
}

func (r reference) Digest() digest.Digest {
	return r.digest
}

type repository struct {
	domain string
	path   string
}

func (r repository) String() string {
	return r.Name()
}

func (r repository) Name() string {
	if r.domain == "" {
		return r.path
	}
	return r.domain + "/" + r.path
}

func (r repository) Domain() string {
	return r.domain
}

func (r repository) Path() string {
	return r.path
}

type digestReference digest.Digest

func (d digestReference) String() string {
	return digest.Digest(d).String()
}

func (d digestReference) Digest() digest.Digest {
	return digest.Digest(d)
}

type taggedReference struct {
	namedRepository
	tag string
}

func (t taggedReference) String() string {
	return t.Name() + ":" + t.tag
}

func (t taggedReference) Tag() string {
	return t.tag
}

type canonicalReference struct {
	namedRepository
	digest digest.Digest
}

func (c canonicalReference) String() string {
	return c.Name() + "@" + c.digest.String()
}

func (c canonicalReference) Digest() digest.Digest {
	return c.digest
}
