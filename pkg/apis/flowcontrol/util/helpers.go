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

package util

import (
	"sort"

	"k8s.io/kubernetes/pkg/apis/flowcontrol"
)

var _ sort.Interface = FlowSchemaSequence{}

// FlowSchemaSequence holds sorted set of pointers to FlowSchema objects.
// FlowSchemaSequence implements `sort.Interface`
type FlowSchemaSequence []*flowcontrol.FlowSchema

func (s FlowSchemaSequence) Len() int {
	return len(s)
}

func (s FlowSchemaSequence) Less(i, j int) bool {
	// the flow-schema w/ lower matching-precedence is prior
	if ip, jp := s[i].Spec.MatchingPrecedence, s[j].Spec.MatchingPrecedence; ip != jp {
		return ip < jp
	}
	// sort alphabetically
	return s[i].Name < s[j].Name
}

func (s FlowSchemaSequence) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
