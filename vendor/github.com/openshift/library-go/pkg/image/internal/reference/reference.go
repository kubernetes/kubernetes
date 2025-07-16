// Package reference provides a general type to represent any way of referencing images within the registry.
// Its main purpose is to abstract tags and digests (content-addressable hash).
//
// Grammar
//
//	reference                       := name [ ":" tag ] [ "@" digest ]
//	name                            := [hostname '/'] component ['/' component]*
//	hostname                        := hostcomponent ['.' hostcomponent]* [':' port-number]
//	hostcomponent                   := /([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])/
//	port-number                     := /[0-9]+/
//	component                       := alpha-numeric [separator alpha-numeric]*
//	alpha-numeric                   := /[a-z0-9]+/
//	separator                       := /[_.]|__|[-]*/
//
//	tag                             := /[\w][\w.-]{0,127}/
//
//	digest                          := digest-algorithm ":" digest-hex
//	digest-algorithm                := digest-algorithm-component [ digest-algorithm-separator digest-algorithm-component ]
//	digest-algorithm-separator      := /[+.-_]/
//	digest-algorithm-component      := /[A-Za-z][A-Za-z0-9]*/
//	digest-hex                      := /[0-9a-fA-F]{32,}/ ; At least 128 bit digest value
package reference

import (
	"errors"
	"fmt"
	"path"
	"strings"

	"github.com/openshift/library-go/pkg/image/internal/digest"
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
// name including a name with hostname and digest
type Canonical interface {
	Named
	Digest() digest.Digest
}

// SplitHostname splits a named reference into a
// hostname and name string. If no valid hostname is
// found, the hostname is empty and the full value
// is returned as name
func SplitHostname(named Named) (string, string) {
	name := named.Name()
	match := anchoredNameRegexp.FindStringSubmatch(name)
	if len(match) != 3 {
		return "", name
	}
	return match[1], match[2]
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

	ref := reference{
		name: matches[1],
		tag:  matches[2],
	}
	if matches[3] != "" {
		var err error
		ref.digest, err = digest.ParseDigest(matches[3])
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
// the Named interface. The reference must have a name, otherwise an error is
// returned.
// If an error was encountered it is returned, along with a nil Reference.
// NOTE: ParseNamed will not handle short digests.
func ParseNamed(s string) (Named, error) {
	ref, err := Parse(s)
	if err != nil {
		return nil, err
	}
	named, isNamed := ref.(Named)
	if !isNamed {
		return nil, fmt.Errorf("reference %s has no name", ref.String())
	}
	return named, nil
}

// WithName returns a named object representing the given string. If the input
// is invalid ErrReferenceInvalidFormat will be returned.
func WithName(name string) (Named, error) {
	if len(name) > NameTotalLengthMax {
		return nil, ErrNameTooLong
	}
	if !anchoredNameRegexp.MatchString(name) {
		return nil, ErrReferenceInvalidFormat
	}
	return repository(name), nil
}

// WithTag combines the name from "name" and the tag from "tag" to form a
// reference incorporating both the name and the tag.
func WithTag(name Named, tag string) (NamedTagged, error) {
	if !anchoredTagRegexp.MatchString(tag) {
		return nil, ErrTagInvalidFormat
	}
	if canonical, ok := name.(Canonical); ok {
		return reference{
			name:   name.Name(),
			tag:    tag,
			digest: canonical.Digest(),
		}, nil
	}
	return taggedReference{
		name: name.Name(),
		tag:  tag,
	}, nil
}

// WithDigest combines the name from "name" and the digest from "digest" to form
// a reference incorporating both the name and the digest.
func WithDigest(name Named, digest digest.Digest) (Canonical, error) {
	if !anchoredDigestRegexp.MatchString(digest.String()) {
		return nil, ErrDigestInvalidFormat
	}
	if tagged, ok := name.(Tagged); ok {
		return reference{
			name:   name.Name(),
			tag:    tagged.Tag(),
			digest: digest,
		}, nil
	}
	return canonicalReference{
		name:   name.Name(),
		digest: digest,
	}, nil
}

// Match reports whether ref matches the specified pattern.
// See https://godoc.org/path#Match for supported patterns.
func Match(pattern string, ref Reference) (bool, error) {
	matched, err := path.Match(pattern, ref.String())
	if namedRef, isNamed := ref.(Named); isNamed && !matched {
		matched, _ = path.Match(pattern, namedRef.Name())
	}
	return matched, err
}

// TrimNamed removes any tag or digest from the named reference.
func TrimNamed(ref Named) Named {
	return repository(ref.Name())
}

func getBestReferenceType(ref reference) Reference {
	if ref.name == "" {
		// Allow digest only references
		if ref.digest != "" {
			return digestReference(ref.digest)
		}
		return nil
	}
	if ref.tag == "" {
		if ref.digest != "" {
			return canonicalReference{
				name:   ref.name,
				digest: ref.digest,
			}
		}
		return repository(ref.name)
	}
	if ref.digest == "" {
		return taggedReference{
			name: ref.name,
			tag:  ref.tag,
		}
	}

	return ref
}

type reference struct {
	name   string
	tag    string
	digest digest.Digest
}

func (r reference) String() string {
	return r.name + ":" + r.tag + "@" + r.digest.String()
}

func (r reference) Name() string {
	return r.name
}

func (r reference) Tag() string {
	return r.tag
}

func (r reference) Digest() digest.Digest {
	return r.digest
}

type repository string

func (r repository) String() string {
	return string(r)
}

func (r repository) Name() string {
	return string(r)
}

type digestReference digest.Digest

func (d digestReference) String() string {
	return string(d)
}

func (d digestReference) Digest() digest.Digest {
	return digest.Digest(d)
}

type taggedReference struct {
	name string
	tag  string
}

func (t taggedReference) String() string {
	return t.name + ":" + t.tag
}

func (t taggedReference) Name() string {
	return t.name
}

func (t taggedReference) Tag() string {
	return t.tag
}

type canonicalReference struct {
	name   string
	digest digest.Digest
}

func (c canonicalReference) String() string {
	return c.name + "@" + c.digest.String()
}

func (c canonicalReference) Name() string {
	return c.name
}

func (c canonicalReference) Digest() digest.Digest {
	return c.digest
}
