// Copyright 2024 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !go1.21

package model

import (
	"fmt"
	"sort"
	"strings"
)

// String was optimized using functions not available for go 1.20
// or lower. We keep the old implementation for compatibility with client_golang.
// Once client golang drops support for go 1.20 (scheduled for August 2024), this
// file can be removed.
func (l LabelSet) String() string {
	labelNames := make([]string, 0, len(l))
	for name := range l {
		labelNames = append(labelNames, string(name))
	}
	sort.Strings(labelNames)
	lstrs := make([]string, 0, len(l))
	for _, name := range labelNames {
		lstrs = append(lstrs, fmt.Sprintf("%s=%q", name, l[LabelName(name)]))
	}
	return fmt.Sprintf("{%s}", strings.Join(lstrs, ", "))
}
