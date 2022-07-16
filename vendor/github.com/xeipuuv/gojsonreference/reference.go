// Copyright 2015 xeipuuv ( https://github.com/xeipuuv )
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

// author  			xeipuuv
// author-github 	https://github.com/xeipuuv
// author-mail		xeipuuv@gmail.com
//
// repository-name	gojsonreference
// repository-desc	An implementation of JSON Reference - Go language
//
// description		Main and unique file.
//
// created      	26-02-2013

package gojsonreference

import (
	"errors"
	"net/url"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/xeipuuv/gojsonpointer"
)

const (
	const_fragment_char = `#`
)

func NewJsonReference(jsonReferenceString string) (JsonReference, error) {

	var r JsonReference
	err := r.parse(jsonReferenceString)
	return r, err

}

type JsonReference struct {
	referenceUrl     *url.URL
	referencePointer gojsonpointer.JsonPointer

	HasFullUrl      bool
	HasUrlPathOnly  bool
	HasFragmentOnly bool
	HasFileScheme   bool
	HasFullFilePath bool
}

func (r *JsonReference) GetUrl() *url.URL {
	return r.referenceUrl
}

func (r *JsonReference) GetPointer() *gojsonpointer.JsonPointer {
	return &r.referencePointer
}

func (r *JsonReference) String() string {

	if r.referenceUrl != nil {
		return r.referenceUrl.String()
	}

	if r.HasFragmentOnly {
		return const_fragment_char + r.referencePointer.String()
	}

	return r.referencePointer.String()
}

func (r *JsonReference) IsCanonical() bool {
	return (r.HasFileScheme && r.HasFullFilePath) || (!r.HasFileScheme && r.HasFullUrl)
}

// "Constructor", parses the given string JSON reference
func (r *JsonReference) parse(jsonReferenceString string) (err error) {

	r.referenceUrl, err = url.Parse(jsonReferenceString)
	if err != nil {
		return
	}
	refUrl := r.referenceUrl

	if refUrl.Scheme != "" && refUrl.Host != "" {
		r.HasFullUrl = true
	} else {
		if refUrl.Path != "" {
			r.HasUrlPathOnly = true
		} else if refUrl.RawQuery == "" && refUrl.Fragment != "" {
			r.HasFragmentOnly = true
		}
	}

	r.HasFileScheme = refUrl.Scheme == "file"
	if runtime.GOOS == "windows" {
		// on Windows, a file URL may have an extra leading slash, and if it
		// doesn't then its first component will be treated as the host by the
		// Go runtime
		if refUrl.Host == "" && strings.HasPrefix(refUrl.Path, "/") {
			r.HasFullFilePath = filepath.IsAbs(refUrl.Path[1:])
		} else {
			r.HasFullFilePath = filepath.IsAbs(refUrl.Host + refUrl.Path)
		}
	} else {
		r.HasFullFilePath = filepath.IsAbs(refUrl.Path)
	}

	// invalid json-pointer error means url has no json-pointer fragment. simply ignore error
	r.referencePointer, _ = gojsonpointer.NewJsonPointer(refUrl.Fragment)

	return
}

// Creates a new reference from a parent and a child
// If the child cannot inherit from the parent, an error is returned
func (r *JsonReference) Inherits(child JsonReference) (*JsonReference, error) {
	if child.GetUrl() == nil {
		return nil, errors.New("childUrl is nil!")
	}

	if r.GetUrl() == nil {
		return nil, errors.New("parentUrl is nil!")
	}

	// Get a copy of the parent url to make sure we do not modify the original.
	// URL reference resolving fails if the fragment of the child is empty, but the parent's is not.
	// The fragment of the child must be used, so the fragment of the parent is manually removed.
	parentUrl := *r.GetUrl()
	parentUrl.Fragment = ""

	ref, err := NewJsonReference(parentUrl.ResolveReference(child.GetUrl()).String())
	if err != nil {
		return nil, err
	}
	return &ref, err
}
