/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

// This flag type is internal to stdlib:
// https://github.com/golang/go/blob/master/src/cmd/internal/obj/flag.go
type int64Value int64

func (i *int64Value) Set(s string) error {
	v, err := strconv.ParseInt(s, 0, 64)
	*i = int64Value(v)
	return err
}

func (i *int64Value) Get() interface{} {
	return int64(*i)
}

func (i *int64Value) String() string {
	return fmt.Sprintf("%v", *i)
}

// NewInt64 behaves as flag.IntVar, but using an int64 type.
func NewInt64(v *int64) flag.Value {
	return (*int64Value)(v)
}

type int64ptrValue struct {
	val **int64
}

func (i *int64ptrValue) Set(s string) error {
	v, err := strconv.ParseInt(s, 0, 64)
	*i.val = new(int64)
	**i.val = int64(v)
	return err
}

func (i *int64ptrValue) Get() interface{} {
	if i.val == nil || *i.val == nil {
		return nil
	}
	return **i.val
}

func (i *int64ptrValue) String() string {
	return fmt.Sprintf("%v", i.Get())
}

func NewOptionalInt64(v **int64) flag.Value {
	return &int64ptrValue{val: v}
}
