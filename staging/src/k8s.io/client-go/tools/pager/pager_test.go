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
	"reflect"
	"testing"

	"golang.org/x/net/context"
	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1alpha1 "k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
)

func list(count int, rv string) *metainternalversion.List {
	var list metainternalversion.List
	for i := 0; i < count; i++ {
		list.Items = append(list.Items, &metav1alpha1.PartialObjectMetadata{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("%d", i),
			},
		})
	}
	list.ResourceVersion = rv
	return &list
}

type testPager struct {
	t          *testing.T
	rv         string
	index      int
	remaining  int
	last       int
	continuing bool
	done       bool
	expectPage int64
}

func (p *testPager) reset() {
	p.continuing = false
	p.remaining += p.index
	p.index = 0
	p.last = 0
	p.done = false
}

func (p *testPager) PagedList(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
	if p.done {
		p.t.Errorf("did not expect additional call to paged list")
		return nil, fmt.Errorf("unexpected list call")
	}
	expectedContinue := fmt.Sprintf("%s:%d", p.rv, p.last)
	if options.Limit != p.expectPage || (p.continuing && options.Continue != expectedContinue) {
		p.t.Errorf("invariant violated, expected limit %d and continue %s, got %#v", p.expectPage, expectedContinue, options)
		return nil, fmt.Errorf("invariant violated")
	}
	var list metainternalversion.List
	total := options.Limit
	if total == 0 {
		total = int64(p.remaining)
	}
	for i := int64(0); i < total; i++ {
		if p.remaining <= 0 {
			break
		}
		list.Items = append(list.Items, &metav1alpha1.PartialObjectMetadata{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("%d", p.index),
			},
		})
		p.remaining--
		p.index++
	}
	p.last = p.index
	if p.remaining > 0 {
		list.Continue = fmt.Sprintf("%s:%d", p.rv, p.last)
		p.continuing = true
	} else {
		p.done = true
	}
	list.ResourceVersion = p.rv
	return &list, nil
}

func (p *testPager) ExpiresOnSecondPage(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
	if p.continuing {
		p.done = true
		return nil, errors.NewResourceExpired("this list has expired")
	}
	return p.PagedList(ctx, options)
}

func (p *testPager) ExpiresOnSecondPageThenFullList(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
	if p.continuing {
		p.reset()
		p.expectPage = 0
		return nil, errors.NewResourceExpired("this list has expired")
	}
	return p.PagedList(ctx, options)
}

func TestListPager_List(t *testing.T) {
	type fields struct {
		PageSize          int64
		PageFn            ListPageFunc
		FullListIfExpired bool
	}
	type args struct {
		ctx     context.Context
		options metav1.ListOptions
	}
	tests := []struct {
		name      string
		fields    fields
		args      args
		want      runtime.Object
		wantErr   bool
		isExpired bool
	}{
		{
			name:   "empty page",
			fields: fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 0, rv: "rv:20"}).PagedList},
			args:   args{},
			want:   list(0, "rv:20"),
		},
		{
			name:   "one page",
			fields: fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 9, rv: "rv:20"}).PagedList},
			args:   args{},
			want:   list(9, "rv:20"),
		},
		{
			name:   "one full page",
			fields: fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 10, rv: "rv:20"}).PagedList},
			args:   args{},
			want:   list(10, "rv:20"),
		},
		{
			name:   "two pages",
			fields: fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 11, rv: "rv:20"}).PagedList},
			args:   args{},
			want:   list(11, "rv:20"),
		},
		{
			name:   "three pages",
			fields: fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 21, rv: "rv:20"}).PagedList},
			args:   args{},
			want:   list(21, "rv:20"),
		},
		{
			name:      "expires on second page",
			fields:    fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 21, rv: "rv:20"}).ExpiresOnSecondPage},
			args:      args{},
			wantErr:   true,
			isExpired: true,
		},
		{
			name: "expires on second page and then lists",
			fields: fields{
				FullListIfExpired: true,
				PageSize:          10,
				PageFn:            (&testPager{t: t, expectPage: 10, remaining: 21, rv: "rv:20"}).ExpiresOnSecondPageThenFullList,
			},
			args: args{},
			want: list(21, "rv:20"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &ListPager{
				PageSize:          tt.fields.PageSize,
				PageFn:            tt.fields.PageFn,
				FullListIfExpired: tt.fields.FullListIfExpired,
			}
			got, err := p.List(tt.args.ctx, tt.args.options)
			if (err != nil) != tt.wantErr {
				t.Errorf("ListPager.List() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.isExpired != errors.IsResourceExpired(err) {
				t.Errorf("ListPager.List() error = %v, isExpired %v", err, tt.isExpired)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ListPager.List() = %v, want %v", got, tt.want)
			}
		})
	}
}
