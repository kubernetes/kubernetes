// Copyright 2016 The etcd Authors
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

package main

import (
	"fmt"
	"strings"
)

func getSameValue(vals map[string]int64) bool {
	var rv int64
	for _, v := range vals {
		if rv == 0 {
			rv = v
		}
		if rv != v {
			return false
		}
	}
	return true
}

func max(n1, n2 int64) int64 {
	if n1 > n2 {
		return n1
	}
	return n2
}

func errsToError(errs []error) error {
	if len(errs) == 0 {
		return nil
	}
	stringArr := make([]string, len(errs))
	for i, err := range errs {
		stringArr[i] = err.Error()
	}
	return fmt.Errorf(strings.Join(stringArr, ", "))
}
