/*
Copyright 2024 The Kubernetes Authors.

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

package clientcmd

import (
	"github.com/imdario/mergo"
)

// recursively merges src into dst:
// - non-pointer struct fields are recursively merged
// - maps are shallow merged with src keys taking priority over dst
// - non-zero src fields encountered during recursion that are not maps or structs overwrite and recursion stops
func merge(dst, src any) error {
	return mergo.Merge(dst, src, mergo.WithOverride)
}
