// Copyright 2013 sigu-399 ( https://github.com/sigu-399 )
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// author       sigu-399
// author-github  https://github.com/sigu-399
// author-mail    sigu.399@gmail.com
//
// repository-name  jsonreference
// repository-desc  An implementation of JSON Reference - Go language
//
// description    Main and unique file.
//
// created        26-02-2013

package jsonreference

import (
	"errors"
	"net/url"
	"strings"

	"github.com/PuerkitoBio/purell"
	"github.com/go-openapi/jsonpointer"
)

const (
	fragmentRune = `#`
)

// New creates a new reference for the given string
func New(jsonReferenceString string) (Ref, error) {

	var r Ref
	err := r.parse(jsonReferenceString)
	return r, err

}

// MustCreateRef parses the ref string and panics when it's invalid.
// Use the New method for a version that returns an error
func MustCreateRef(ref string) Ref {
	r, err := New(ref)
	if err != nil {
		panic(err)
	}
	return r
}

// Ref represents a json reference object
type Ref struct {
	referenceURL     *url.URL
	referencePointer jsonpointer.Pointer

	HasFullURL      bool
	HasURLPathOnly  bool
	HasFragmentOnly bool
	HasFileScheme   bool
	HasFullFilePath bool
}

// GetURL gets the URL for this reference
func (r *Ref) GetURL() *url.URL {
	return r.referenceURL
}

// GetPointer gets the json pointer for this reference
func (r *Ref) GetPointer() *jsonpointer.Pointer {
	return &r.referencePointer
}

// String returns the best version of the url for this reference
func (r *Ref) String() string {

	if r.referenceURL != nil {
		return r.referenceURL.String()
	}

	if r.HasFragmentOnly {
		return fragmentRune + r.referencePointer.String()
	}

	return r.referencePointer.String()
}

// IsRoot returns true if this reference is a root document
func (r *Ref) IsRoot() bool {
	return r.referenceURL != nil &&
		!r.IsCanonical() &&
		!r.HasURLPathOnly &&
		r.referenceURL.Fragment == ""
}

// IsCanonical returns true when this pointer starts with http(s):// or file://
func (r *Ref) IsCanonical() bool {
	return (r.HasFileScheme && r.HasFullFilePath) || (!r.HasFileScheme && r.HasFullURL)
}

// "Constructor", parses the given string JSON reference
func (r *Ref) parse(jsonReferenceString string) error {

	parsed, err := url.Parse(jsonReferenceString)
	if err != nil {
		return err
	}

	r.referenceURL, _ = url.Parse(purell.NormalizeURL(parsed, purell.FlagsSafe|purell.FlagRemoveDuplicateSlashes))
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

// Inherits creates a new reference from a parent and a child
// If the child cannot inherit from the parent, an error is returned
func (r *Ref) Inherits(child Ref) (*Ref, error) {
	childURL := child.GetURL()
	parentURL := r.GetURL()
	if childURL == nil {
		return nil, errors.New("child url is nil")
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
