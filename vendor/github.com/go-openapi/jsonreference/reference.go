// SPDX-FileCopyrightText: Copyright (c) 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package jsonreference

import (
	"errors"
	"net/url"
	"strings"

	"github.com/go-openapi/jsonpointer"
	"github.com/go-openapi/jsonreference/internal"
)

const (
	fragmentRune = `#`
)

var ErrChildURL = errors.New("child url is nil")

// Ref represents a json reference object.
type Ref struct {
	referenceURL     *url.URL
	referencePointer jsonpointer.Pointer

	HasFullURL      bool
	HasURLPathOnly  bool
	HasFragmentOnly bool
	HasFileScheme   bool
	HasFullFilePath bool
}

// New creates a new reference for the given string.
func New(jsonReferenceString string) (Ref, error) {
	var r Ref
	err := r.parse(jsonReferenceString)
	return r, err
}

// MustCreateRef parses the ref string and panics when it's invalid.
// Use the New method for a version that returns an error.
func MustCreateRef(ref string) Ref {
	r, err := New(ref)
	if err != nil {
		panic(err)
	}

	return r
}

// GetURL gets the URL for this reference.
func (r *Ref) GetURL() *url.URL {
	return r.referenceURL
}

// GetPointer gets the json pointer for this reference.
func (r *Ref) GetPointer() *jsonpointer.Pointer {
	return &r.referencePointer
}

// String returns the best version of the url for this reference.
func (r *Ref) String() string {
	if r.referenceURL != nil {
		return r.referenceURL.String()
	}

	if r.HasFragmentOnly {
		return fragmentRune + r.referencePointer.String()
	}

	return r.referencePointer.String()
}

// IsRoot returns true if this reference is a root document.
func (r *Ref) IsRoot() bool {
	return r.referenceURL != nil &&
		!r.IsCanonical() &&
		!r.HasURLPathOnly &&
		r.referenceURL.Fragment == ""
}

// IsCanonical returns true when this pointer starts with http(s):// or file://.
func (r *Ref) IsCanonical() bool {
	return (r.HasFileScheme && r.HasFullFilePath) || (!r.HasFileScheme && r.HasFullURL)
}

// Inherits creates a new reference from a parent and a child
// If the child cannot inherit from the parent, an error is returned.
func (r *Ref) Inherits(child Ref) (*Ref, error) {
	childURL := child.GetURL()
	parentURL := r.GetURL()
	if childURL == nil {
		return nil, ErrChildURL
	}
	if parentURL == nil {
		return &child, nil
	}

	ref, err := New(parentURL.ResolveReference(childURL).String())
	if err != nil {
		return nil, err
	}
	return &ref, nil
}

// "Constructor", parses the given string JSON reference.
func (r *Ref) parse(jsonReferenceString string) error {
	parsed, err := url.Parse(jsonReferenceString)
	if err != nil {
		return err
	}

	internal.NormalizeURL(parsed)

	r.referenceURL = parsed
	refURL := r.referenceURL

	if refURL.Scheme != "" && refURL.Host != "" {
		r.HasFullURL = true
	} else {
		if refURL.Path != "" {
			r.HasURLPathOnly = true
		} else if refURL.RawQuery == "" && refURL.Fragment != "" {
			r.HasFragmentOnly = true
		}
	}

	r.HasFileScheme = refURL.Scheme == "file"
	r.HasFullFilePath = strings.HasPrefix(refURL.Path, "/")

	// invalid json-pointer error means url has no json-pointer fragment. simply ignore error
	r.referencePointer, _ = jsonpointer.New(refURL.Fragment)

	return nil
}
