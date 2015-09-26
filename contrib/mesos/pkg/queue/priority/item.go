/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package priority

type Indexable interface {
	Index() int
	SetIndex(int)
}

type Item interface {
	Indexable
	Value() interface{}
	Priority() Priority
}

type item struct {
	value    interface{}
	priority Priority
	index    int
}

func NewItem(value interface{}, priority Priority) *item {
	return &item{
		value:    value,
		priority: priority,
		index:    -1,
	}
}

func (i *item) Value() interface{} {
	return i.value
}

func (i *item) Priority() Priority {
	return i.priority
}

func (i *item) Index() int {
	return i.index
}

func (i *item) SetIndex(index int) {
	i.index = index
}
