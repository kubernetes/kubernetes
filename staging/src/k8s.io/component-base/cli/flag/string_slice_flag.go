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

package flag

import (
	goflag "flag"
	"fmt"
	"strings"

	"github.com/spf13/pflag"
)

// StringSlice implements goflag.Value and plfag.Value,
// and allows set to be invoked repeatedly to accumulate values.
type StringSlice struct {
	value   *[]string
	changed bool
}

func NewStringSlice(s *[]string) *StringSlice {
	return &StringSlice{value: s}
}

var _ goflag.Value = &StringSlice{}
var _ pflag.Value = &StringSlice{}

func (s *StringSlice) String() string {
	if s == nil || s.value == nil {
		return ""
	}
	return strings.Join(*s.value, " ")
}

func (s *StringSlice) Set(val string) error {
	if s.value == nil {
		return fmt.Errorf("no target (nil pointer to []string)")
	}
	if !s.changed {
		*s.value = make([]string, 0)
	}
	*s.value = append(*s.value, val)
	s.changed = true
	return nil
}

func (StringSlice) Type() string {
	return "sliceString"
}
