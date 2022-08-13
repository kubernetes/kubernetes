// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package spec

import (
	"net/url"
	"path"
	"strings"
)

const fileScheme = "file"

// normalizeURI ensures that all $ref paths used internally by the expander are canonicalized.
//
// NOTE(windows): there is a tolerance over the strict URI format on windows.
//
// The normalizer accepts relative file URLs like 'Path\File.JSON' as well as absolute file URLs like
// 'C:\Path\file.Yaml'.
//
// Both are canonicalized with a "file://" scheme, slashes and a lower-cased path:
// 'file:///c:/path/file.yaml'
//
// URLs can be specified with a file scheme, like in 'file:///folder/file.json' or
// 'file:///c:\folder\File.json'.
//
// URLs like file://C:\folder are considered invalid (i.e. there is no host 'c:\folder') and a "repair"
// is attempted.
//
// The base path argument is assumed to be canonicalized (e.g. using normalizeBase()).
func normalizeURI(refPath, base string) string {
	refURL, err := url.Parse(refPath)
	if err != nil {
		specLogger.Printf("warning: invalid URI in $ref  %q: %v", refPath, err)
		refURL, refPath = repairURI(refPath)
	}

	fixWindowsURI(refURL, refPath) // noop on non-windows OS

	refURL.Path = path.Clean(refURL.Path)
	if refURL.Path == "." {
		refURL.Path = ""
	}

	r := MustCreateRef(refURL.String())
	if r.IsCanonical() {
		return refURL.String()
	}

	baseURL, _ := url.Parse(base)
	if path.IsAbs(refURL.Path) {
		baseURL.Path = refURL.Path
	} else if refURL.Path != "" {
		baseURL.Path = path.Join(path.Dir(baseURL.Path), refURL.Path)
	}
	// copying fragment from ref to base
	baseURL.Fragment = refURL.Fragment

	return baseURL.String()
}

// denormalizeRef returns the simplest notation for a normalized $ref, given the path of the original root document.
//
// When calling this, we assume that:
// * $ref is a canonical URI
// * originalRelativeBase is a canonical URI
//
// denormalizeRef is currently used when we rewrite a $ref after a circular $ref has been detected.
// In this case, expansion stops and normally renders the internal canonical $ref.
//
// This internal $ref is eventually rebased to the original RelativeBase used for the expansion.
//
// There is a special case for schemas that are anchored with an "id":
// in that case, the rebasing is performed // against the id only if this is an anchor for the initial root document.
// All other intermediate "id"'s found along the way are ignored for the purpose of rebasing.
//
func denormalizeRef(ref *Ref, originalRelativeBase, id string) Ref {
	debugLog("denormalizeRef called:\n$ref: %q\noriginal: %s\nroot ID:%s", ref.String(), originalRelativeBase, id)

	if ref.String() == "" || ref.IsRoot() || ref.HasFragmentOnly {
		// short circuit: $ref to current doc
		return *ref
	}

	if id != "" {
		idBaseURL, err := url.Parse(id)
		if err == nil { // if the schema id is not usable as a URI, ignore it
			if ref, ok := rebase(ref, idBaseURL, true); ok { // rebase, but keep references to root unchaged (do not want $ref: "")
				// $ref relative to the ID of the schema in the root document
				return ref
			}
		}
	}

	originalRelativeBaseURL, _ := url.Parse(originalRelativeBase)

	r, _ := rebase(ref, originalRelativeBaseURL, false)

	return r
}

func rebase(ref *Ref, v *url.URL, notEqual bool) (Ref, bool) {
	var newBase url.URL

	u := ref.GetURL()

	if u.Scheme != v.Scheme || u.Host != v.Host {
		return *ref, false
	}

	docPath := v.Path
	v.Path = path.Dir(v.Path)

	if v.Path == "." {
		v.Path = ""
	} else if !strings.HasSuffix(v.Path, "/") {
		v.Path += "/"
	}

	newBase.Fragment = u.Fragment

	if strings.HasPrefix(u.Path, docPath) {
		newBase.Path = strings.TrimPrefix(u.Path, docPath)
	} else {
		newBase.Path = strings.TrimPrefix(u.Path, v.Path)
	}

	if notEqual && newBase.Path == "" && newBase.Fragment == "" {
		// do not want rebasing to end up in an empty $ref
		return *ref, false
	}

	if path.IsAbs(newBase.Path) {
		// whenever we end up with an absolute path, specify the scheme and host
		newBase.Scheme = v.Scheme
		newBase.Host = v.Host
	}

	return MustCreateRef(newBase.String()), true
}

// normalizeRef canonicalize a Ref, using a canonical relativeBase as its absolute anchor
func normalizeRef(ref *Ref, relativeBase string) *Ref {
	r := MustCreateRef(normalizeURI(ref.String(), relativeBase))
	return &r
}

// normalizeBase performs a normalization of the input base path.
//
// This always yields a canonical URI (absolute), usable for the document cache.
//
// It ensures that all further internal work on basePath may safely assume
// a non-empty, cross-platform, canonical URI (i.e. absolute).
//
// This normalization tolerates windows paths (e.g. C:\x\y\File.dat) and transform this
// in a file:// URL with lower cased drive letter and path.
//
// See also: https://en.wikipedia.org/wiki/File_URI_scheme
func normalizeBase(in string) string {
	u, err := url.Parse(in)
	if err != nil {
		specLogger.Printf("warning: invalid URI in RelativeBase  %q: %v", in, err)
		u, in = repairURI(in)
	}

	u.Fragment = "" // any fragment in the base is irrelevant

	fixWindowsURI(u, in) // noop on non-windows OS

	u.Path = path.Clean(u.Path)
	if u.Path == "." { // empty after Clean()
		u.Path = ""
	}

	if u.Scheme != "" {
		if path.IsAbs(u.Path) || u.Scheme != fileScheme {
			// this is absolute or explicitly not a local file: we're good
			return u.String()
		}
	}

	// no scheme or file scheme with relative path: assume file and make it absolute
	// enforce scheme file://... with absolute path.
	//
	// If the input path is relative, we anchor the path to the current working directory.
	// NOTE: we may end up with a host component. Leave it unchanged: e.g. file://host/folder/file.json

	u.Scheme = fileScheme
	u.Path = absPath(u.Path) // platform-dependent
	u.RawQuery = ""          // any query component is irrelevant for a base
	return u.String()
}
