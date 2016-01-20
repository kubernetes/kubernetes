/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package util

import (
	"fmt"
	"strconv"
)

// BoolFlag is a boolean flag compatible with flags and pflags that keeps track of whether it had a value supplied or not.
// Getting this flag to act like a normal bool, where true/false are not required needs a little bit of extra code, example:
//    f := cmd.Flags().VarPF(&BoolFlagVar, "flagname", "", "help about the flag")
//    f.NoOptDefVal = "true"
type BoolFlag struct {
	// If Set has been invoked this value is true
	provided bool
	// The exact value provided on the flag
	value bool
}

func (f *BoolFlag) Default(value bool) {
	f.value = value
}

func (f BoolFlag) String() string {
	return fmt.Sprintf("%t", f.value)
}

func (f BoolFlag) Value() bool {
	return f.value
}

func (f *BoolFlag) Set(value string) error {
	boolVal, err := strconv.ParseBool(value)
	if err != nil {
		return err
	}

	f.value = boolVal
	f.provided = true

	return nil
}

func (f BoolFlag) Provided() bool {
	return f.provided
}

func (f *BoolFlag) Type() string {
	return "bool"
}
