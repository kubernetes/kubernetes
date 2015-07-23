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
)

/* NoSpliStringList is just a list of strings.
 * Unlike util.StringList, it does not interpret the values in any way and it
 * does not split the values by ','.
 * After this code, the list contains {'a,b', 'c'}:
 * list NoSplitStringList; list.Set('a,b'); list.Set('c')
 */
type NoSplitStringList []string

func (sl *NoSplitStringList) String() string {
	return fmt.Sprint(*sl)
}

func (sl *NoSplitStringList) Set(value string) error {
	*sl = append(*sl, value)
	return nil
}

func (*NoSplitStringList) Type() string {
	return "noSplitStringList"
}
