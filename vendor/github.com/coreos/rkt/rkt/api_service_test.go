// Copyright 2015 The rkt Authors
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
	"io/ioutil"
	"net"
	"os"
	"testing"

	"github.com/coreos/rkt/api/v1alpha"
	"github.com/coreos/rkt/pkg/log"
)

func TestFilterPod(t *testing.T) {
	tests := []struct {
		pod    *v1alpha.Pod
		filter *v1alpha.PodFilter
		result bool
	}{
		// Has the status.
		{
			&v1alpha.Pod{
				State: v1alpha.PodState_POD_STATE_RUNNING,
			},
			&v1alpha.PodFilter{
				States: []v1alpha.PodState{v1alpha.PodState_POD_STATE_RUNNING},
			},
			true,
		},
		// Doesn't have the status.
		{
			&v1alpha.Pod{
				State: v1alpha.PodState_POD_STATE_EXITED,
			},
			&v1alpha.PodFilter{
				States: []v1alpha.PodState{v1alpha.PodState_POD_STATE_RUNNING},
			},
			false,
		},
		// Has all app names.
		{
			&v1alpha.Pod{
				Apps: []*v1alpha.App{
					{Name: "app-foo"},
					{Name: "app-bar"},
				},
			},
			&v1alpha.PodFilter{
				AppNames: []string{"app-foo", "app-bar"},
			},
			true,
		},
		// Doesn't have all app name.
		{
			&v1alpha.Pod{
				Apps: []*v1alpha.App{
					{Name: "app-foo"},
					{Name: "app-bar"},
				},
			},
			&v1alpha.PodFilter{
				AppNames: []string{"app-foo", "app-bar", "app-baz"},
			},
			false,
		},
		// Has all network names.
		{
			&v1alpha.Pod{
				Networks: []*v1alpha.Network{
					{Name: "network-foo"},
					{Name: "network-bar"},
				},
			},
			&v1alpha.PodFilter{
				NetworkNames: []string{"network-foo", "network-bar"},
			},
			true,
		},
		// Doesn't have all network names.
		{
			&v1alpha.Pod{
				Networks: []*v1alpha.Network{
					{Name: "network-foo"},
					{Name: "network-bar"},
				},
			},
			&v1alpha.PodFilter{
				NetworkNames: []string{"network-foo", "network-bar", "network-baz"},
			},
			false,
		},
		// Has all annotations.
		{
			&v1alpha.Pod{
				Annotations: []*v1alpha.KeyValue{
					{"annotation-key-foo", "annotation-value-foo"},
					{"annotation-key-bar", "annotation-value-bar"},
				},
			},
			&v1alpha.PodFilter{
				Annotations: []*v1alpha.KeyValue{
					{"annotation-key-foo", "annotation-value-foo"},
					{"annotation-key-bar", "annotation-value-bar"},
				},
			},
			true,
		},
		// Doesn't have all annotation keys.
		{
			&v1alpha.Pod{
				Annotations: []*v1alpha.KeyValue{
					{"annotation-key-foo", "annotation-value-foo"},
					{"annotation-key-bar", "annotation-value-bar"},
				},
			},
			&v1alpha.PodFilter{
				Annotations: []*v1alpha.KeyValue{
					{"annotation-key-foo", "annotation-value-foo"},
					{"annotation-key-bar", "annotation-value-bar"},
					{"annotation-key-baz", "annotation-value-baz"},
				},
			},
			false,
		},
		// Doesn't have all annotation values.
		{
			&v1alpha.Pod{
				Annotations: []*v1alpha.KeyValue{
					{"annotation-key-foo", "annotation-value-foo"},
					{"annotation-key-bar", "annotation-value-bar"},
				},
			},
			&v1alpha.PodFilter{
				Annotations: []*v1alpha.KeyValue{
					{"annotation-key-foo", "annotation-value-foo"},
					{"annotation-key-bar", "annotation-value-baz"},
				},
			},
			false,
		},
		// Doesn't satisfy any filter conditions.
		{
			&v1alpha.Pod{
				Apps:        []*v1alpha.App{{Name: "app-foo"}},
				Networks:    []*v1alpha.Network{{Name: "network-foo"}},
				Annotations: []*v1alpha.KeyValue{{"annotation-key-foo", "annotation-value-foo"}},
			},
			&v1alpha.PodFilter{
				AppNames:     []string{"app-bar"},
				NetworkNames: []string{"network-bar"},
				Annotations:  []*v1alpha.KeyValue{{"annotation-key-foo", "annotation-value-bar"}},
			},
			false,
		},
		// Satisfies some filter conditions.
		{
			&v1alpha.Pod{
				Apps:        []*v1alpha.App{{Name: "app-foo"}},
				Networks:    []*v1alpha.Network{{Name: "network-foo"}},
				Annotations: []*v1alpha.KeyValue{{"annotation-key-foo", "annotation-value-foo"}},
			},
			&v1alpha.PodFilter{
				AppNames:     []string{"app-foo", "app-bar"},
				NetworkNames: []string{"network-bar"},
				Annotations:  []*v1alpha.KeyValue{{"annotation-key-bar", "annotation-value-bar"}},
			},
			false,
		},
		// Satisfies all filter conditions.
		{
			&v1alpha.Pod{
				Apps:        []*v1alpha.App{{Name: "app-foo"}},
				Networks:    []*v1alpha.Network{{Name: "network-foo"}},
				Annotations: []*v1alpha.KeyValue{{"annotation-key-foo", "annotation-value-foo"}},
			},
			&v1alpha.PodFilter{
				AppNames:     []string{"app-foo"},
				NetworkNames: []string{"network-foo"},
				Annotations:  []*v1alpha.KeyValue{{"annotation-key-foo", "annotation-value-foo"}},
			},
			true,
		},
	}

	for i, tt := range tests {
		result := satisfiesPodFilter(*tt.pod, *tt.filter)
		if result != tt.result {
			t.Errorf("#%d: got %v, want %v", i, result, tt.result)
		}
	}
}

func TestFilterPodAny(t *testing.T) {
	tests := []struct {
		pod     *v1alpha.Pod
		filters []*v1alpha.PodFilter
		result  bool
	}{
		// No filters.
		{
			&v1alpha.Pod{
				Apps:     []*v1alpha.App{{Name: "app-foo"}},
				Networks: []*v1alpha.Network{{Name: "network-foo"}},
			},
			nil,
			true,
		},
		// Satisfies all filters.
		{
			&v1alpha.Pod{
				Apps:     []*v1alpha.App{{Name: "app-foo"}},
				Networks: []*v1alpha.Network{{Name: "network-foo"}},
			},
			[]*v1alpha.PodFilter{
				{AppNames: []string{"app-foo"}},
				{NetworkNames: []string{"network-foo"}},
			},
			true,
		},
		// Satisfies any filters.
		{
			&v1alpha.Pod{
				Apps:     []*v1alpha.App{{Name: "app-foo"}},
				Networks: []*v1alpha.Network{{Name: "network-foo"}},
			},
			[]*v1alpha.PodFilter{
				{AppNames: []string{"app-foo"}},
				{NetworkNames: []string{"network-bar"}},
			},
			true,
		},
		// Satisfies none filters.
		{
			&v1alpha.Pod{
				Apps:     []*v1alpha.App{{Name: "app-foo"}},
				Networks: []*v1alpha.Network{{Name: "network-foo"}},
			},
			[]*v1alpha.PodFilter{
				{AppNames: []string{"app-bar"}},
				{NetworkNames: []string{"network-bar"}},
			},
			false,
		},
	}

	for i, tt := range tests {
		result := satisfiesAnyPodFilters(tt.pod, tt.filters)
		if result != tt.result {
			t.Errorf("#%d: got %v, want %v", i, result, tt.result)
		}
	}
}

func TestFilterImage(t *testing.T) {
	tests := []struct {
		image  *v1alpha.Image
		filter *v1alpha.ImageFilter
		result bool
	}{
		// Has the id.
		{
			&v1alpha.Image{
				Id: "id-foo",
			},
			&v1alpha.ImageFilter{
				Ids: []string{"id-first", "id-foo", "id-last"},
			},
			true,
		},
		// Doesn't have the id.
		{
			&v1alpha.Image{
				Id: "id-foo",
			},
			&v1alpha.ImageFilter{
				Ids: []string{"id-first", "id-second", "id-last"},
			},
			false,
		},
		// Has the prefix in the name.
		{
			&v1alpha.Image{
				Name: "prefix-foo-foo",
			},
			&v1alpha.ImageFilter{
				Prefixes: []string{"prefix-first", "prefix-foo", "prefix-last"},
			},
			true,
		},
		// Doesn't have the prefix in the name.
		{
			&v1alpha.Image{
				Name: "prefix-foo-foo",
			},
			&v1alpha.ImageFilter{
				Prefixes: []string{"prefix-first", "prefix-second", "prefix-last"},
			},
			false,
		},
		// Has the base name in the name.
		{
			&v1alpha.Image{
				Name: "foo/basename-foo",
			},
			&v1alpha.ImageFilter{
				BaseNames: []string{"basename-first", "basename-foo", "basename-last"},
			},
			true,
		},
		// Doesn't have the base name in the name.
		{
			&v1alpha.Image{
				Name: "foo/basename-foo",
			},
			&v1alpha.ImageFilter{
				BaseNames: []string{"basename-first", "basename-second", "basename-last"},
			},
			false,
		},
		// Has the keyword in the name.
		{
			&v1alpha.Image{
				Name: "foo-keyword-foo-foo",
			},
			&v1alpha.ImageFilter{
				Keywords: []string{"keyword-first", "keyword-foo", "keyword-last"},
			},
			true,
		},
		// Doesn't have the keyword in the name.
		{
			&v1alpha.Image{
				Name: "foo-keyword-foo-foo",
			},
			&v1alpha.ImageFilter{
				Keywords: []string{"keyword-first", "keyword-second", "keyword-last"},
			},
			false,
		},
		// Has all the labels in the manifest.
		{
			&v1alpha.Image{
				Labels: []*v1alpha.KeyValue{
					{"label-key-foo", "label-value-foo"},
					{"label-key-bar", "label-value-bar"},
				},
			},
			&v1alpha.ImageFilter{
				Labels: []*v1alpha.KeyValue{
					{"label-key-foo", "label-value-foo"},
					{"label-key-bar", "label-value-bar"},
				},
			},
			true,
		},
		// Doesn't have all the label keys in the manifest.
		{
			&v1alpha.Image{
				Labels: []*v1alpha.KeyValue{
					{"label-key-foo", "label-value-foo"},
					{"label-key-bar", "label-value-bar"},
				},
			},
			&v1alpha.ImageFilter{
				Labels: []*v1alpha.KeyValue{
					{"label-key-foo", "label-value-foo"},
					{"label-key-bar", "label-value-bar"},
					{"label-key-baz", "label-value-baz"},
				},
			},
			false,
		},
		// Doesn't have all the label values in the manifest.
		{
			&v1alpha.Image{
				Labels: []*v1alpha.KeyValue{
					{"label-key-foo", "label-value-foo"},
					{"label-key-bar", "label-value-bar"},
				},
			},
			&v1alpha.ImageFilter{
				Labels: []*v1alpha.KeyValue{
					{"label-key-foo", "label-value-foo"},
					{"label-key-bar", "label-value-baz"},
				},
			},
			false,
		},
		// Has all the annotation in the manifest.
		{
			&v1alpha.Image{
				Annotations: []*v1alpha.KeyValue{
					{"annotation-key-foo", "annotation-value-foo"},
					{"annotation-key-bar", "annotation-value-bar"},
				},
			},
			&v1alpha.ImageFilter{
				Annotations: []*v1alpha.KeyValue{
					{"annotation-key-foo", "annotation-value-foo"},
					{"annotation-key-bar", "annotation-value-bar"},
				},
			},
			true,
		},
		// Doesn't have all the annotation keys in the manifest.
		{
			&v1alpha.Image{
				Annotations: []*v1alpha.KeyValue{
					{"annotation-key-foo", "annotation-value-foo"},
					{"annotation-key-bar", "annotation-value-bar"},
				},
			},
			&v1alpha.ImageFilter{
				Annotations: []*v1alpha.KeyValue{
					{"annotation-key-foo", "annotation-value-foo"},
					{"annotation-key-bar", "annotation-value-bar"},
					{"annotation-key-baz", "annotation-value-baz"},
				},
			},
			false,
		},
		// Doesn't have all the annotation values in the manifest.
		{
			&v1alpha.Image{
				Annotations: []*v1alpha.KeyValue{
					{"annotation-key-foo", "annotation-value-foo"},
					{"annotation-key-bar", "annotation-value-bar"},
				},
			},
			&v1alpha.ImageFilter{
				Annotations: []*v1alpha.KeyValue{
					{"annotation-key-foo", "annotation-value-foo"},
					{"annotation-key-bar", "annotation-value-baz"},
				},
			},
			false,
		},
		// Satisfies 'imported after'.
		{
			&v1alpha.Image{ImportTimestamp: 1024},
			&v1alpha.ImageFilter{
				ImportedAfter: 1023,
			},
			true,
		},
		// Doesn't satisfy 'imported after'.
		{
			&v1alpha.Image{ImportTimestamp: 1024},
			&v1alpha.ImageFilter{
				ImportedAfter: 1024,
			},
			false,
		},
		// Satisfies 'imported before'.
		{
			&v1alpha.Image{ImportTimestamp: 1024},
			&v1alpha.ImageFilter{
				ImportedBefore: 1025,
			},
			true,
		},
		// Doesn't satisfy 'imported before'.
		{
			&v1alpha.Image{ImportTimestamp: 1024},
			&v1alpha.ImageFilter{
				ImportedBefore: 1024,
			},
			false,
		},
		// Match one of the full names.
		{
			&v1alpha.Image{Name: "foo/basename-foo"},
			&v1alpha.ImageFilter{
				FullNames: []string{"foo/basename-foo", "foo/basename-bar"},
			},
			true,
		},
		// Doesn't match any full names.
		{
			&v1alpha.Image{Name: "foo/basename-foo"},
			&v1alpha.ImageFilter{
				FullNames: []string{"bar/basename-foo", "foo/basename-bar"},
			},
			false,
		},
		// Doesn't satisfy any filter conditions.
		{
			&v1alpha.Image{
				Id:              "id-foo",
				Name:            "prefix-foo-keyword-foo/basename-foo",
				Version:         "1.0",
				ImportTimestamp: 1024,
				Labels:          []*v1alpha.KeyValue{{"label-key-foo", "label-value-foo"}},
				Annotations:     []*v1alpha.KeyValue{{"annotation-key-foo", "annotation-value-foo"}},
			},
			&v1alpha.ImageFilter{
				Ids:            []string{"id-bar"},
				Prefixes:       []string{"prefix-bar"},
				BaseNames:      []string{"basename-bar"},
				Keywords:       []string{"keyword-bar"},
				Labels:         []*v1alpha.KeyValue{{"label-key-bar", "label-value-bar"}},
				Annotations:    []*v1alpha.KeyValue{{"annotation-key-bar", "annotation-value-bar"}},
				ImportedBefore: 1024,
				ImportedAfter:  1024,
			},
			false,
		},
		// Satisfies some filter conditions.
		{
			&v1alpha.Image{
				Id:              "id-foo",
				Name:            "prefix-foo-keyword-foo/basename-foo",
				Version:         "1.0",
				ImportTimestamp: 1024,
				Labels:          []*v1alpha.KeyValue{{"label-key-foo", "label-value-foo"}},
				Annotations:     []*v1alpha.KeyValue{{"annotation-key-foo", "annotation-value-foo"}},
			},
			&v1alpha.ImageFilter{
				Ids:            []string{"id-bar", "id-foo"},
				Prefixes:       []string{"prefix-bar"},
				BaseNames:      []string{"basename-bar"},
				Keywords:       []string{"keyword-bar"},
				Labels:         []*v1alpha.KeyValue{{"label-key-bar", "label-value-bar"}},
				Annotations:    []*v1alpha.KeyValue{{"annotation-key-bar", "annotation-value-bar"}},
				ImportedBefore: 1024,
				ImportedAfter:  1024,
			},
			false,
		},
		// Satisfies all filter conditions.
		{
			&v1alpha.Image{
				Id:              "id-foo",
				Name:            "prefix-foo-keyword-foo/basename-foo",
				Version:         "1.0",
				ImportTimestamp: 1024,
				Labels:          []*v1alpha.KeyValue{{"label-key-foo", "label-value-foo"}},
				Annotations:     []*v1alpha.KeyValue{{"annotation-key-foo", "annotation-value-foo"}},
			},
			&v1alpha.ImageFilter{
				Ids:            []string{"id-bar", "id-foo"},
				Prefixes:       []string{"prefix-bar", "prefix-foo"},
				BaseNames:      []string{"basename-bar", "basename-foo"},
				Keywords:       []string{"keyword-bar", "keyword-foo"},
				Labels:         []*v1alpha.KeyValue{{"label-key-foo", "label-value-foo"}},
				Annotations:    []*v1alpha.KeyValue{{"annotation-key-foo", "annotation-value-foo"}},
				ImportedBefore: 1025,
				ImportedAfter:  1023,
			},
			true,
		},
	}

	for i, tt := range tests {
		result := satisfiesImageFilter(*tt.image, *tt.filter)
		if result != tt.result {
			t.Errorf("#%d: got %v, want %v", i, result, tt.result)
		}
	}
}

func TestFilterImageAny(t *testing.T) {
	tests := []struct {
		image   *v1alpha.Image
		filters []*v1alpha.ImageFilter
		result  bool
	}{
		// No filters.
		{
			&v1alpha.Image{
				Id:   "id-foo",
				Name: "prefix-foo-keyword-foo/basename-foo",
			},
			nil,
			true,
		},
		// Satisfies all filters.
		{
			&v1alpha.Image{
				Id:   "id-foo",
				Name: "prefix-foo-keyword-foo/basename-foo",
			},
			[]*v1alpha.ImageFilter{
				{Ids: []string{"id-foo"}},
				{BaseNames: []string{"basename-foo"}},
			},
			true,
		},
		// Satisfies any filters.
		{
			&v1alpha.Image{
				Id:   "id-foo",
				Name: "prefix-foo-keyword-foo/basename-foo",
			},
			[]*v1alpha.ImageFilter{
				{Ids: []string{"id-foo"}},
				{BaseNames: []string{"basename-bar"}},
			},
			true,
		},
		// Satisfies none filters.
		{
			&v1alpha.Image{
				Id:   "id-foo",
				Name: "prefix-foo-keyword-foo/basename-foo",
			},
			[]*v1alpha.ImageFilter{
				{Ids: []string{"id-bar"}},
				{BaseNames: []string{"basename-bar"}},
			},
			false,
		},
	}

	for i, tt := range tests {
		result := satisfiesAnyImageFilters(tt.image, tt.filters)
		if result != tt.result {
			t.Errorf("#%d: got %v, want %v", i, result, tt.result)
		}
	}
}

// Test that we open the correct kinds of sockets
func TestOpenSocket(t *testing.T) {
	stderr = log.New(os.Stderr, "TestOpenSocket", globalFlags.Debug)
	// get a temp unix socket for us to play with
	tempdir, err := ioutil.TempDir("", "TestOpenSocket")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempdir)

	l, err := net.Listen("unix", tempdir+"/sock")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	lfd, err := l.(*net.UnixListener).File()
	if err != nil {
		t.Fatal(err)
	}
	defer lfd.Close()

	// Mock out the systemd FD function
	systemdFDs = func(x bool) []*os.File {
		return []*os.File{lfd}
	}

	// Test that we will open a systemd socket when asked for
	l1, err := openAPISockets()
	if err != nil {
		t.Fatal(err)
	}
	if len(l1) != 1 {
		t.Errorf("expected len(l1) = 1, got %d", len(l1))
	}

	// Test that we fail when --listen is passed with a systemd socket
	flagAPIServiceListenAddr = "localhost:0"
	_, err = openAPISockets()
	if err == nil {
		t.Error("openAPISockets() did not fail when passed systemd socket and --listen")
	}

	// Then, disable socket mode and ask it to open a random tcp server port
	systemdFDs = func(x bool) []*os.File {
		return nil
	}

	l2, err := openAPISockets()
	if err != nil {
		t.Fatal("failed to open socket", err)
	}

	for _, ll := range l2 {
		defer ll.Close()
	}

	if len(l2) != 1 {
		t.Errorf("expected len(l2) == 1, but got %d", len(l2))
	}

	switch l2[0].(type) {
	case *net.TCPListener:
	// ok
	default:
		t.Errorf("expected type=*net.TCPListener, got %T", l2[0])
	}
}
