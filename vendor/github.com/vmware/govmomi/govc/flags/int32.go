/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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
type int32Value int32

func (i *int32Value) Set(s string) error {
	v, err := strconv.ParseInt(s, 0, 64)
	*i = int32Value(v)
	return err
}

func (i *int32Value) Get() interface{} {
	return int32(*i)
}

func (i *int32Value) String() string {
	return fmt.Sprintf("%v", *i)
}

// NewInt32 behaves as flag.IntVar, but using an int32 type.
func NewInt32(v *int32) flag.Value {
	return (*int32Value)(v)
}
