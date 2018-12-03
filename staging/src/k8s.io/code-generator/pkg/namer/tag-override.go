/*
Copyright 2016 The Kubernetes Authors.

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

package namer

import (
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
)

// TagOverrideNamer is a namer which pulls names from a given tag, if specified,
// and otherwise falls back to a different namer.
type TagOverrideNamer struct {
	tagName  string
	fallback namer.Namer
}

// Name returns the tag value if it exists. It no tag was found the fallback namer will be used
func (n *TagOverrideNamer) Name(t *types.Type) string {
	if nameOverride := extractTag(n.tagName, append(t.SecondClosestCommentLines, t.CommentLines...)); nameOverride != "" {
		return nameOverride
	}

	return n.fallback.Name(t)
}

// NewTagOverrideNamer creates a namer.Namer which uses the contents of the given tag as
// the name, or falls back to another Namer if the tag is not present.
func NewTagOverrideNamer(tagName string, fallback namer.Namer) namer.Namer {
	return &TagOverrideNamer{
		tagName:  tagName,
		fallback: fallback,
	}
}

// extractTag gets the comment-tags for the key.  If the tag did not exist, it
// returns the empty string.
func extractTag(key string, lines []string) string {
	val, present := types.ExtractCommentTags("+", lines)[key]
	if !present || len(val) < 1 {
		return ""
	}

	return val[0]
}
