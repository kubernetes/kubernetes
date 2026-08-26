/*
Copyright The Kubernetes Authors.

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

package correctness

import (
	"context"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/storage"
)

// NewEmptyModel returns a new Model with no items.
func NewEmptyModel(prefix string) *Model {
	return &Model{
		Prefix:          prefix,
		ResourceVersion: 1,
		Items:           make(map[string]runtime.Object),
	}
}

// Model is a state machine that mimics kubernetes storage behavior. Used for Linarizability testing of storage.
type Model struct {
	Items           map[string]runtime.Object
	ResourceVersion uint64
	Prefix          string
}

func (s *Model) Clone() *Model {
	clone := &Model{
		Items:           make(map[string]runtime.Object, len(s.Items)),
		ResourceVersion: s.ResourceVersion,
		Prefix:          s.Prefix,
	}
	for k, v := range s.Items {
		if v != nil {
			clone.Items[k] = v.DeepCopyObject()
		}
	}
	return clone
}

func (s *Model) Equal(other *Model) bool {
	return reflect.DeepEqual(s, other)
}

// Step applies an operation to the sequential state machine.
func (s *Model) Step(input Request, output Response) (ok bool, next *Model) {
	next = s
	var expected Response
	switch input.Op {
	case OpCreate:
		next = s.Clone()
		expected = next.create(input.Key, input.Object)
	case OpDelete:
		next = s.Clone()
		expected = next.delete(context.Background(), input.Key, input.Preconditions, nil)
	case OpGet:
		expected = s.get(input.Key, input.GetOptions)
	}
	if !reflect.DeepEqual(expected, output) {
		return false, s
	}
	return true, next
}

func (s *Model) create(key string, obj runtime.Object) Response {
	if _, exists := s.Items[key]; exists {
		return Response{Object: nil, Err: storage.NewKeyExistsError(s.prepareKey(key), 0)}
	}
	s.ResourceVersion++
	copied := obj.DeepCopyObject()
	accessor, err := meta.Accessor(copied)
	if err != nil {
		return Response{Object: nil, Err: err}
	}
	accessor.SetResourceVersion(strconv.FormatUint(s.ResourceVersion, 10))
	s.Items[key] = copied
	return Response{Object: copied, Err: nil}
}

func (s *Model) get(key string, opts storage.GetOptions) Response {
	stored, exists := s.Items[key]
	if !exists {
		if opts.IgnoreNotFound {
			return Response{Object: nil, Err: nil}
		}
		return Response{Object: nil, Err: storage.NewKeyNotFoundError(s.prepareKey(key), 0)}
	}
	return Response{Object: stored.DeepCopyObject(), Err: nil}
}

func (s *Model) delete(ctx context.Context, key string, preconditions *storage.Preconditions, validateDeletion storage.ValidateObjectFunc) Response {
	stored, exists := s.Items[key]
	if !exists {
		return Response{Object: nil, Err: storage.NewKeyNotFoundError(s.prepareKey(key), 0)}
	}
	if err := preconditions.Check(s.prepareKey(key), stored); err != nil {
		return Response{Object: nil, Err: err}
	}
	if validateDeletion != nil && stored != nil {
		if err := validateDeletion(ctx, stored); err != nil {
			return Response{Object: nil, Err: err}
		}
	}
	s.ResourceVersion++
	deletedObj := stored.DeepCopyObject()
	accessor, err := meta.Accessor(deletedObj)
	if err != nil {
		return Response{Object: nil, Err: err}
	}
	accessor.SetResourceVersion(strconv.FormatUint(s.ResourceVersion, 10))
	delete(s.Items, key)
	return Response{Object: deletedObj, Err: nil}
}

func (s *Model) Describe() string {
	items := []string{}
	for key, item := range s.Items {
		accessor, err := meta.Accessor(item)
		if err != nil {
			items = append(items, fmt.Sprintf("<p>%s: err: %v</p>", key, err))
		} else {
			items = append(items, fmt.Sprintf("<p>%s: %s, RV:%s</p>", key, accessor.GetUID(), accessor.GetResourceVersion()))
		}
	}
	return fmt.Sprintf("RV: %d, Items: %s", s.ResourceVersion, strings.Join(items, ""))
}

func (s *Model) prepareKey(key string) string {
	if s.Prefix == "" {
		return key
	}
	p := s.Prefix
	if !strings.HasPrefix(p, "/") {
		p = "/" + p
	}
	p = strings.TrimSuffix(p, "/")
	if !strings.HasPrefix(key, "/") {
		key = "/" + key
	}
	return p + key
}
