/*
Copyright 2021 The Kubernetes Authors.

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

package generators

import (
	"k8s.io/code-generator/cmd/client-gen/generators/util"
)

// Tags represents a genclient configuration for a single type.
type Tags struct {
	util.Tags
}

func (t Tags) NeedsInformerInjection() bool {
	return t.GenerateClient && !t.NoVerbs && t.HasVerb("list") && t.HasVerb("watch")
}

// MustParseClientGenTags calls ParseClientGenTags but instead of returning error it panics.
func MustParseClientGenTags(lines []string) Tags {
	ret := Tags{
		Tags: util.MustParseClientGenTags(lines),
	}

	return ret
}
