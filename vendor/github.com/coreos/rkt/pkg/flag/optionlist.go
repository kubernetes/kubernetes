// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package flag

import (
	"errors"
	"fmt"
	"strings"

	"github.com/hashicorp/errwrap"
)

// OptionList is a flag value type supporting a csv list of options
type OptionList struct {
	// Options field holds all specified and valid options. Should
	// not be modified.
	Options     []string
	allOptions  []string
	permissible map[string]struct{}
	typeName    string
}

// NewOptionList initializes an OptionList. PermissibleOptions is the complete
// set of allowable options. It will set all options specified in defaultOptions
// as provided; they will all be cleared if this flag is passed in the CLI
func NewOptionList(permissibleOptions []string, defaultOptions string) (*OptionList, error) {
	permissible := make(map[string]struct{})
	ol := &OptionList{
		allOptions:  permissibleOptions,
		permissible: permissible,
		typeName:    "OptionList",
	}

	for _, o := range permissibleOptions {
		ol.permissible[o] = struct{}{}
	}

	if err := ol.Set(defaultOptions); err != nil {
		return nil, errwrap.Wrap(errors.New("problem setting defaults"), err)
	}

	return ol, nil
}

func (ol *OptionList) Set(s string) error {
	ol.Options = nil
	if s == "" {
		return nil
	}
	options := strings.Split(strings.ToLower(s), ",")
	seen := map[string]struct{}{}
	for _, o := range options {
		if _, ok := ol.permissible[o]; !ok {
			return fmt.Errorf("unknown option %q", o)
		}
		if _, ok := seen[o]; ok {
			return fmt.Errorf("duplicated option %q", o)
		}
		ol.Options = append(ol.Options, o)
		seen[o] = struct{}{}
	}

	return nil
}

func (ol *OptionList) String() string {
	return strings.Join(ol.Options, ",")
}

func (ol *OptionList) Type() string {
	return ol.typeName
}

func (ol *OptionList) PermissibleString() string {
	return fmt.Sprintf(`"%s"`, strings.Join(ol.allOptions, `", "`))
}
