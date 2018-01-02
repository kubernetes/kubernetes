/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package flags

import (
	"flag"
	"fmt"
	"strconv"
)

type optionalBool struct {
	val **bool
}

func (b *optionalBool) Set(s string) error {
	v, err := strconv.ParseBool(s)
	*b.val = &v
	return err
}

func (b *optionalBool) Get() interface{} {
	if *b.val == nil {
		return nil
	}
	return **b.val
}

func (b *optionalBool) String() string {
	if b.val == nil || *b.val == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%v", **b.val)
}

func (b *optionalBool) IsBoolFlag() bool { return true }

// NewOptionalBool returns a flag.Value implementation where there is no default value.
// This avoids sending a default value over the wire as using flag.BoolVar() would.
func NewOptionalBool(v **bool) flag.Value {
	return &optionalBool{v}
}
