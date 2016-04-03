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

// bitFlags is a flag value type supporting a csv list of options stored as bits
type bitFlags struct {
	*OptionList
	flags   int
	flagMap map[string]int
}

func newBitFlags(permissibleOptions []string, defaultOptions string, flagMap map[string]int) (*bitFlags, error) {
	ol, err := NewOptionList(permissibleOptions, defaultOptions)
	if err != nil {
		return nil, err
	}

	bf := &bitFlags{
		OptionList: ol,
		flagMap:    flagMap,
	}
	bf.typeName = "bitFlags"

	if err := bf.Set(defaultOptions); err != nil {
		return nil, errwrap.Wrap(errors.New("problem setting defaults"), err)
	}

	return bf, nil
}

func (bf *bitFlags) Set(s string) error {
	if err := bf.OptionList.Set(s); err != nil {
		return err
	}
	bf.flags = 0
	for _, o := range bf.Options {
		if b, ok := bf.flagMap[o]; ok {
			bf.flags |= b
		} else {
			return fmt.Errorf("couldn't find flag for %v", o)
		}
	}
	return nil
}

func (bf *bitFlags) hasFlag(f int) bool {
	return (bf.flags & f) == f
}
