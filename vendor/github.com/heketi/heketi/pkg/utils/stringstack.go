//
// Copyright (c) 2015 The heketi Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package utils

type StringStack struct {
	list []string
}

func NewStringStack() *StringStack {
	a := &StringStack{}
	a.list = make([]string, 0)
	return a
}

func (a *StringStack) IsEmpty() bool {
	return len(a.list) == 0
}

func (a *StringStack) Pop() (x string) {
	x, a.list = a.list[0], a.list[1:len(a.list)]
	return
}

func (a *StringStack) Push(x string) {
	a.list = append(a.list, x)
}
