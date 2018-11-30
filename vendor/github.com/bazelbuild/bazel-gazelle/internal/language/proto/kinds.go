/* Copyright 2018 The Bazel Authors. All rights reserved.

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

package proto

import "github.com/bazelbuild/bazel-gazelle/internal/rule"

var protoKinds = map[string]rule.KindInfo{
	"proto_library": {
		NonEmptyAttrs:  map[string]bool{"srcs": true},
		MergeableAttrs: map[string]bool{"srcs": true},
		ResolveAttrs:   map[string]bool{"deps": true},
	},
}

func (_ *protoLang) Kinds() map[string]rule.KindInfo { return protoKinds }
func (_ *protoLang) Loads() []rule.LoadInfo          { return nil }
