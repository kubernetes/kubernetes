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

package generators

import (
	"k8s.io/gengo/types"
)

// extractTag gets the comment-tags for the key.  If the tag did not exist, it
// returns the empty string.
func extractTag(key string, lines []string) string {
	val, present := types.ExtractCommentTags("+", lines)[key]
	if !present || len(val) < 1 {
		return ""
	}

	return val[0]
}
