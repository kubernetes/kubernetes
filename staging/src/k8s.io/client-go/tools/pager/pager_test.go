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
	"context"
	"fmt"
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
)

func list(count int, rv string) *metainternalversion.List {
	var list metainternalversion.List
	for i := 0; i < count; i++ {
		list.Items = append(list.Items, &metav1beta1.PartialObjectMetadata{
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
	if options.Continue != "" && options.ResourceVersion != "" {
		p.t.Errorf("invariant violated, specifying resource version (%s) is not allowed when using continue (%s).", options.ResourceVersion, options.Continue)
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
		list.Items = append(list.Items, &metav1beta1.PartialObjectMetadata{
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
		{
			name:   "two pages with resourceVersion",
			fields: fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 11, rv: "rv:20"}).PagedList},
			args:   args{options: metav1.ListOptions{ResourceVersion: "rv:10"}},
			want:   list(11, "rv:20"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &ListPager{
				PageSize:          tt.fields.PageSize,
				PageFn:            tt.fields.PageFn,
				FullListIfExpired: tt.fields.FullListIfExpired,
			}
			ctx := tt.args.ctx
			if ctx == nil {
				ctx = context.Background()
			}
			got, err := p.List(ctx, tt.args.options)
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

func TestListPager_EachListItem(t *testing.T) {
	type fields struct {
		PageSize int64
		PageFn   ListPageFunc
	}
	tests := []struct {
		name                 string
		fields               fields
		want                 runtime.Object
		wantErr              bool
		wantPanic            bool
		isExpired            bool
		processorErrorOnItem int
		processorPanicOnItem int
		cancelContextOnItem  int
	}{
		{
			name:   "empty page",
			fields: fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 0, rv: "rv:20"}).PagedList},
			want:   list(0, "rv:20"),
		},
		{
			name:   "one page",
			fields: fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 9, rv: "rv:20"}).PagedList},
			want:   list(9, "rv:20"),
		},
		{
			name:   "one full page",
			fields: fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 10, rv: "rv:20"}).PagedList},
			want:   list(10, "rv:20"),
		},
		{
			name:   "two pages",
			fields: fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 11, rv: "rv:20"}).PagedList},
			want:   list(11, "rv:20"),
		},
		{
			name:   "three pages",
			fields: fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 21, rv: "rv:20"}).PagedList},
			want:   list(21, "rv:20"),
		},
		{
			name:      "expires on second page",
			fields:    fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 21, rv: "rv:20"}).ExpiresOnSecondPage},
			want:      list(10, "rv:20"), // all items on the first page should have been visited
			wantErr:   true,
			isExpired: true,
		},
		{
			name:                 "error processing item",
			fields:               fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 51, rv: "rv:20"}).PagedList},
			want:                 list(3, "rv:20"), // all the items <= the one the processor returned an error on should have been visited
			wantPanic:            true,
			processorPanicOnItem: 3,
		},
		{
			name:                "cancel context while processing",
			fields:              fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 51, rv: "rv:20"}).PagedList},
			want:                list(3, "rv:20"), // all the items <= the one the processor returned an error on should have been visited
			wantErr:             true,
			cancelContextOnItem: 3,
		},
		{
			name:      "panic processing item",
			fields:    fields{PageSize: 10, PageFn: (&testPager{t: t, expectPage: 10, remaining: 51, rv: "rv:20"}).PagedList},
			want:      list(3, "rv:20"), // all the items <= the one the processor returned an error on should have been visited
			wantPanic: true,
		},
	}

	processorErr := fmt.Errorf("processor error")
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			p := &ListPager{
				PageSize: tt.fields.PageSize,
				PageFn:   tt.fields.PageFn,
			}
			var items []runtime.Object

			fn := func(obj runtime.Object) error {
				items = append(items, obj)
				if tt.processorErrorOnItem > 0 && len(items) == tt.processorErrorOnItem {
					return processorErr
				}
				if tt.processorPanicOnItem > 0 && len(items) == tt.processorPanicOnItem {
					panic(processorErr)
				}
				if tt.cancelContextOnItem > 0 && len(items) == tt.cancelContextOnItem {
					cancel()
				}
				return nil
			}
			var err error
			var panic interface{}
			func() {
				defer func() {
					panic = recover()
				}()
				err = p.EachListItem(ctx, metav1.ListOptions{}, fn)
			}()
			if (panic != nil) && !tt.wantPanic {
				t.Fatalf(".EachListItem() panic = %v, wantPanic %v", panic, tt.wantPanic)
			} else {
				return
			}
			if (err != nil) != tt.wantErr {
				t.Errorf("ListPager.EachListItem() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.isExpired != errors.IsResourceExpired(err) {
				t.Errorf("ListPager.EachListItem() error = %v, isExpired %v", err, tt.isExpired)
				return
			}
			if tt.processorErrorOnItem > 0 && err != processorErr {
				t.Errorf("ListPager.EachListItem() error = %v, processorErrorOnItem %d", err, tt.processorErrorOnItem)
				return
			}
			l := tt.want.(*metainternalversion.List)
			if !reflect.DeepEqual(items, l.Items) {
				t.Errorf("ListPager.EachListItem() = %v, want %v", items, l.Items)
			}
		})
	}
}

func TestListPager_eachListPageBuffered(t *testing.T) {
	tests := []struct {
		name           string
		totalPages     int
		pagesProcessed int
		wantPageLists  int
		pageBufferSize int32
		pageSize       int
	}{
		{
			name:           "no buffer, one total page",
			totalPages:     1,
			pagesProcessed: 1,
			wantPageLists:  1,
			pageBufferSize: 0,
		}, {
			name:           "no buffer, 1/5 pages processed",
			totalPages:     5,
			pagesProcessed: 1,
			wantPageLists:  2, // 1 received for processing, 1 listed
			pageBufferSize: 0,
		},
		{
			name:           "no buffer, 2/5 pages processed",
			totalPages:     5,
			pagesProcessed: 2,
			wantPageLists:  3,
			pageBufferSize: 0,
		},
		{
			name:           "no buffer, 5/5 pages processed",
			totalPages:     5,
			pagesProcessed: 5,
			wantPageLists:  5,
			pageBufferSize: 0,
		},
		{
			name:           "size 1 buffer, 1/5 pages processed",
			totalPages:     5,
			pagesProcessed: 1,
			wantPageLists:  3,
			pageBufferSize: 1,
		},
		{
			name:           "size 1 buffer, 5/5 pages processed",
			totalPages:     5,
			pagesProcessed: 5,
			wantPageLists:  5,
			pageBufferSize: 1,
		},
		{
			name:           "size 10 buffer, 1/5 page processed",
			totalPages:     5,
			pagesProcessed: 1,
			wantPageLists:  5,
			pageBufferSize: 10, // buffer is larger than list
		},
	}
	processorErr := fmt.Errorf("processor error")
	pageSize := 10
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pgr := &testPager{t: t, expectPage: int64(pageSize), remaining: tt.totalPages * pageSize, rv: "rv:20"}
			pageLists := 0
			wantedPageListsDone := make(chan struct{})
			listFn := func(ctx context.Context, options metav1.ListOptions) (runtime.Object, error) {
				pageLists++
				if pageLists == tt.wantPageLists {
					close(wantedPageListsDone)
				}
				return pgr.PagedList(ctx, options)
			}
			p := &ListPager{
				PageSize:       int64(pageSize),
				PageBufferSize: tt.pageBufferSize,
				PageFn:         listFn,
			}

			pagesProcessed := 0
			fn := func(obj runtime.Object) error {
				pagesProcessed++
				if tt.pagesProcessed == pagesProcessed && tt.wantPageLists > 0 {
					// wait for buffering to catch up
					select {
					case <-time.After(time.Second):
						return fmt.Errorf("Timed out waiting for %d page lists", tt.wantPageLists)
					case <-wantedPageListsDone:
					}
					return processorErr
				}
				return nil
			}
			err := p.eachListChunkBuffered(context.Background(), metav1.ListOptions{}, fn)
			if tt.pagesProcessed > 0 && err == processorErr {
				// expected
			} else if err != nil {
				t.Fatal(err)
			}
			if tt.wantPageLists > 0 && pageLists != tt.wantPageLists {
				t.Errorf("expected %d page lists, got %d", tt.wantPageLists, pageLists)
			}
			if pagesProcessed != tt.pagesProcessed {
				t.Errorf("expected %d pages processed, got %d", tt.pagesProcessed, pagesProcessed)
			}
		})
	}
}
