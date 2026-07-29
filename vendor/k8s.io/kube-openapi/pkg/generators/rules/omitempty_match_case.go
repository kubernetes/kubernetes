/*
Copyright 2018 The Kubernetes Authors.

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

package rules

import (
	"reflect"
	"strings"

	"k8s.io/gengo/v2/types"
)

// OmitEmptyMatchCase implements APIRule interface.
// "omitempty" must appear verbatim (no case variants).
type OmitEmptyMatchCase struct{}

func (n *OmitEmptyMatchCase) Name() string {
	return "omitempty_match_case"
}

func (n *OmitEmptyMatchCase) Validate(t *types.Type) ([]string, error) {
	fields := make([]string, 0)

	// Only validate struct type and ignore the rest
	switch t.Kind {
	case types.Struct:
		for _, m := range t.Members {
			goName := m.Name
			jsonTag, ok := reflect.StructTag(m.Tags).Lookup("json")
			if !ok {
				continue
			}

			parts := strings.Split(jsonTag, ",")
			if len(parts) < 2 {
				// no tags other than name
				continue
			}
			if parts[0] == "-" {
				// not serialized
				continue
			}
			for _, part := range parts[1:] {
				if strings.EqualFold(part, "omitempty") && part != "omitempty" {
					fields = append(fields, goName)
				}
			}
		}
	}
	return fields, nil
}
