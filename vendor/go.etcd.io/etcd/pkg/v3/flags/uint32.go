// Copyright 2022 The etcd Authors
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

package flags

import (
	"flag"
	"strconv"
)

type uint32Value uint32

// NewUint32Value creates an uint32 instance with the provided value.
func NewUint32Value(v uint32) *uint32Value {
	val := new(uint32Value)
	*val = uint32Value(v)
	return val
}

// Set parses a command line uint32 value.
// Implements "flag.Value" interface.
func (i *uint32Value) Set(s string) error {
	v, err := strconv.ParseUint(s, 0, 32)
	*i = uint32Value(v)
	return err
}

func (i *uint32Value) String() string { return strconv.FormatUint(uint64(*i), 10) }

// Uint32FromFlag return the uint32 value of a flag with the given name
func Uint32FromFlag(fs *flag.FlagSet, name string) uint32 {
	val := *fs.Lookup(name).Value.(*uint32Value)
	return uint32(val)
}
