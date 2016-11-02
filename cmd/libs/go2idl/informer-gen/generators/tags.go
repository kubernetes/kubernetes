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
	"github.com/golang/glog"
	"k8s.io/gengo/types"
)

// extractBoolTagOrDie gets the comment-tags for the key and asserts that, if
// it exists, the value is boolean.  If the tag did not exist, it returns
// false.
func extractBoolTagOrDie(key string, lines []string) bool {
	val, err := types.ExtractSingleBoolCommentTag("+", key, false, lines)
	if err != nil {
		glog.Fatalf(err.Error())
	}
	return val
}
