/* Copyright 2016 The Bazel Authors. All rights reserved.

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

package resolve

import (
	"github.com/bazelbuild/bazel-gazelle/internal/label"
)

// vendoredResolver resolves external packages as packages in vendor/.
type vendoredResolver struct {
	l *label.Labeler
}

var _ nonlocalResolver = (*vendoredResolver)(nil)

func newVendoredResolver(l *label.Labeler) *vendoredResolver {
	return &vendoredResolver{l}
}

func (v *vendoredResolver) resolve(importpath string) (label.Label, error) {
	return v.l.LibraryLabel("vendor/" + importpath), nil
}
