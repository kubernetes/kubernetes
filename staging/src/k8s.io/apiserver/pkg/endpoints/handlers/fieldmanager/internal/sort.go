/*
Copyright 2019 The Kubernetes Authors.

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

package internal

import (
	"sort"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type managedFieldsLessFunc func(p, q metav1.ManagedFieldsEntry) bool

type managedFieldsSorter struct {
	fields []metav1.ManagedFieldsEntry
	less   []managedFieldsLessFunc
}

func (s *managedFieldsSorter) Sort(fields []metav1.ManagedFieldsEntry) {
	s.fields = fields
	sort.Sort(s)
}

// Len is the amount of managedFields to sort
func (s *managedFieldsSorter) Len() int {
	return len(s.fields)
}

// Swap is part of sort.Interface.
func (s *managedFieldsSorter) Swap(p, q int) {
	s.fields[p], s.fields[q] = s.fields[q], s.fields[p]
}

// Less is part of sort.Interface
func (s *managedFieldsSorter) Less(p, q int) bool {
	a, b := s.fields[p], s.fields[q]
	var k int
	for k = 0; k < len(s.less)-1; k++ {
		less := s.less[k]
		switch {
		case less(a, b):
			return true
		case less(b, a):
			return false
		}
	}
	return s.less[k](a, b)
}
