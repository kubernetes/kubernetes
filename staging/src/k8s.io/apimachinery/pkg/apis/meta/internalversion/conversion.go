/*
Copyright The Kubernetes Authors.

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

package internalversion

import (
	"fmt"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/sharding"
)

// Convert_v1_ListOptions_To_internalversion_ListOptions handles conversion from
// the wire-format v1.ListOptions to the internal ListOptions, including shard
// selector parsing.
func Convert_v1_ListOptions_To_internalversion_ListOptions(in *v1.ListOptions, out *ListOptions, s conversion.Scope) error {
	if err := autoConvert_v1_ListOptions_To_internalversion_ListOptions(in, out, s); err != nil {
		return err
	}

	// Parse the new selector field into a ShardSelector if set.
	if in.Selector != "" {
		sel, err := sharding.Parse(in.Selector)
		if err != nil {
			return fmt.Errorf("invalid shard selector: %w", err)
		}
		out.ShardSelector = sel
	}

	return nil
}

// Convert_internalversion_ListOptions_To_v1_ListOptions handles conversion from
// internal ListOptions to the wire-format v1.ListOptions.
func Convert_internalversion_ListOptions_To_v1_ListOptions(in *ListOptions, out *v1.ListOptions, s conversion.Scope) error {
	if err := autoConvert_internalversion_ListOptions_To_v1_ListOptions(in, out, s); err != nil {
		return err
	}

	if in.ShardSelector != nil && !in.ShardSelector.Empty() {
		out.Selector = in.ShardSelector.String()
	}

	return nil
}
