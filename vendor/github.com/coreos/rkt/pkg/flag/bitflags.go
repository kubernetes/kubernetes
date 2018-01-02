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

	"github.com/hashicorp/errwrap"
)

// BitFlags is a flag value type supporting a csv list of options stored as bits
type BitFlags struct {
	*OptionList
	Flags   int
	FlagMap map[string]int
}

// NewBitFlags initializes a simple bitflag version of the OptionList Type.
// flagMap is the mapping from option names to the integer value
func NewBitFlags(permissibleOptions []string, defaultOptions string, flagMap map[string]int) (*BitFlags, error) {
	ol, err := NewOptionList(permissibleOptions, defaultOptions)
	if err != nil {
		return nil, err
	}

	bf := &BitFlags{
		OptionList: ol,
		FlagMap:    flagMap,
	}
	bf.typeName = "BitFlags"

	if err := bf.Set(defaultOptions); err != nil {
		return nil, errwrap.Wrap(errors.New("problem setting defaults"), err)
	}

	return bf, nil
}

func (bf *BitFlags) Set(s string) error {
	if err := bf.OptionList.Set(s); err != nil {
		return err
	}
	bf.Flags = 0
	for _, o := range bf.Options {
		if b, ok := bf.FlagMap[o]; ok {
			bf.Flags |= b
		} else {
			return fmt.Errorf("couldn't find flag for %v", o)
		}
	}
	return nil
}

func (bf *BitFlags) HasFlag(f int) bool {
	return (bf.Flags & f) == f
}
