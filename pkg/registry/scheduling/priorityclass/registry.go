/*
Copyright 2017 The Kubernetes Authors.

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

package priorityclass

import (
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/scheduling"
)

// Registry is an interface for things that know how to store PriorityClass.
type Registry interface {
	ListPriorityClasses(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*scheduling.PriorityClassList, error)
	CreatePriorityClass(ctx genericapirequest.Context, pc *scheduling.PriorityClass) error
	UpdatePriorityClass(ctx genericapirequest.Context, pc *scheduling.PriorityClass) error
	GetPriorityClass(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*scheduling.PriorityClass, error)
	DeletePriorityClass(ctx genericapirequest.Context, name string) error
	WatchPriorityClasses(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
}

// storage puts strong typing around storage calls
type storage struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched
// types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListPriorityClasses(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*scheduling.PriorityClassList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}

	return obj.(*scheduling.PriorityClassList), nil
}

func (s *storage) CreatePriorityClass(ctx genericapirequest.Context, pc *scheduling.PriorityClass) error {
	_, err := s.Create(ctx, pc, false)
	return err
}

func (s *storage) UpdatePriorityClass(ctx genericapirequest.Context, pc *scheduling.PriorityClass) error {
	_, _, err := s.Update(ctx, pc.Name, rest.DefaultUpdatedObjectInfo(pc, api.Scheme))
	return err
}

func (s *storage) WatchPriorityClasses(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetPriorityClass(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*scheduling.PriorityClass, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*scheduling.PriorityClass), nil
}

func (s *storage) DeletePriorityClass(ctx genericapirequest.Context, name string) error {
	_, _, err := s.Delete(ctx, name, nil)
	return err
}
