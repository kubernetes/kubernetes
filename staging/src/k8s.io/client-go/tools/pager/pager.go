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

package pager

import (
	"fmt"

	"golang.org/x/net/context"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

const defaultPageSize = 500

// ListPageFunc returns a list object for the given list options.
type ListPageFunc func(ctx context.Context, opts metav1.ListOptions) (runtime.Object, error)

// SimplePageFunc adapts a context-less list function into one that accepts a context.
func SimplePageFunc(fn func(opts metav1.ListOptions) (runtime.Object, error)) ListPageFunc {
	return func(ctx context.Context, opts metav1.ListOptions) (runtime.Object, error) {
		return fn(opts)
	}
}

// ListPager assists client code in breaking large list queries into multiple
// smaller chunks of PageSize or smaller. PageFn is expected to accept a
// metav1.ListOptions that supports paging and return a list. The pager does
// not alter the field or label selectors on the initial options list.
type ListPager struct {
	PageSize int64
	PageFn   ListPageFunc

	FullListIfExpired bool
}

// New creates a new pager from the provided pager function using the default
// options.
func New(fn ListPageFunc) *ListPager {
	return &ListPager{
		PageSize:          defaultPageSize,
		PageFn:            fn,
		FullListIfExpired: true,
	}
}

// List returns a single list object, but attempts to retrieve smaller chunks from the
// server to reduce the impact on the server. If the chunk attempt fails, it will load
// the full list instead.
func (p *ListPager) List(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
	if options.Limit == 0 {
		options.Limit = p.PageSize
	}
	var list *metainternalversion.List
	for {
		obj, err := p.PageFn(ctx, options)
		if err != nil {
			if !errors.IsResourceExpired(err) || !p.FullListIfExpired {
				return nil, err
			}
			// the list expired while we were processing, fall back to a full list
			options.Limit = 0
			options.Continue = ""
			return p.PageFn(ctx, options)
		}
		m, err := meta.ListAccessor(obj)
		if err != nil {
			return nil, fmt.Errorf("returned object must be a list: %v", err)
		}

		// exit early and return the object we got if we haven't processed any pages
		if len(m.GetContinue()) == 0 && list == nil {
			return obj, nil
		}

		// initialize the list and fill its contents
		if list == nil {
			list = &metainternalversion.List{Items: make([]runtime.Object, 0, options.Limit+1)}
			list.ResourceVersion = m.GetResourceVersion()
			list.SelfLink = m.GetSelfLink()
		}
		if err := meta.EachListItem(obj, func(obj runtime.Object) error {
			list.Items = append(list.Items, obj)
			return nil
		}); err != nil {
			return nil, err
		}

		// if we have no more items, return the list
		if len(m.GetContinue()) == 0 {
			return list, nil
		}

		// set the next loop up
		options.Continue = m.GetContinue()
	}
}
