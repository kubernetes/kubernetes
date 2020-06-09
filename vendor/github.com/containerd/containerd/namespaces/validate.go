/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

// Package namespaces provides tools for working with namespaces across
// containerd.
//
// Namespaces collect resources such as containers and images, into a unique
// identifier space. This means that two applications can use the same
// identifiers and not conflict while using containerd.
//
// This package can be used to ensure that client and server functions
// correctly store the namespace on the context.
package namespaces

import (
	"regexp"

	"github.com/containerd/containerd/errdefs"
	"github.com/pkg/errors"
)

const (
	maxLength = 76
	alpha     = `[A-Za-z]`
	alphanum  = `[A-Za-z0-9]+`
	label     = alpha + alphanum + `(:?[-]+` + alpha + alphanum + `)*`
)

var (
	// namespaceRe validates that a namespace matches valid identifiers.
	//
	// Rules for domains, defined in RFC 1035, section 2.3.1, are used for
	// namespaces.
	namespaceRe = regexp.MustCompile(reAnchor(label + reGroup("[.]"+reGroup(label)) + "*"))
)

// Validate returns nil if the string s is a valid namespace.
//
// To allow such namespace identifiers to be used across various contexts
// safely, the character set has been restricted to that defined for domains in
// RFC 1035, section 2.3.1. This will make namespace identifiers safe for use
// across networks, filesystems and other media.
//
// The identifier specification departs from RFC 1035 in that it allows
// "labels" to start with number and only enforces a total length restriction
// of 76 characters.
//
// While the character set may be expanded in the future, namespace identifiers
// are guaranteed to be safely used as filesystem path components.
//
// For the most part, this doesn't need to be called directly when using the
// context-oriented functions.
func Validate(s string) error {
	if len(s) > maxLength {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "namespace %q greater than maximum length (%d characters)", s, maxLength)
	}

	if !namespaceRe.MatchString(s) {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "namespace %q must match %v", s, namespaceRe)
	}
	return nil
}

func reGroup(s string) string {
	return `(?:` + s + `)`
}

func reAnchor(s string) string {
	return `^` + s + `$`
}
