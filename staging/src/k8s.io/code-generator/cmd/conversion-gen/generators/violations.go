/*
Copyright 2026 The Kubernetes Authors.

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
	"fmt"
	"os"
	"sort"
	"strings"
)

const (
	// hubMemoryIdentityRule: a +k8s:hubType type has a field that is not
	// memory-identical to its internal peer.
	hubMemoryIdentityRule = "hub_memory_identity"
	// hubTypeMissingRule: an internal type has not matching +k8s:hubType.
	hubTypeMissingRule = "hub_type_missing"
)

type violation struct {
	rule     string
	pkg      string
	typeName string
	field    string
}

func (v violation) String() string {
	return fmt.Sprintf("%s%s,%s,%s,%s", "API rule violation: ", v.rule, v.pkg, v.typeName, v.field)
}

func (v violation) less(o violation) bool {
	if v.rule != o.rule {
		return v.rule < o.rule
	}
	if v.pkg != o.pkg {
		return v.pkg < o.pkg
	}
	if v.typeName != o.typeName {
		return v.typeName < o.typeName
	}
	return v.field < o.field
}

// writeViolationReport writes the sorted violations to the report file path.
func writeViolationReport(path string, violations []violation) error {
	sort.Slice(violations, func(i, j int) bool { return violations[i].less(violations[j]) })
	var b strings.Builder
	for _, v := range violations {
		b.WriteString(v.String())
		b.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		return fmt.Errorf("writing conversion-gen violation report %s: %w", path, err)
	}
	return nil
}
