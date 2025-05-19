/*
Copyright 2024 The Kubernetes Authors.

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

package operation

import "testing"

func TestRequest_SubresourceIn(t *testing.T) {
	tests := []struct {
		name                string
		requestSubresources []string
		matchSubresources   []string
		want                bool
	}{
		{
			name:                "subresource match",
			requestSubresources: []string{"x"},
			matchSubresources:   []string{"/x"},
			want:                true,
		},
		{
			name:                "subresource no match",
			requestSubresources: []string{"x"},
			matchSubresources:   []string{"/y"},
			want:                false,
		},
		{
			name:                "root match",
			requestSubresources: []string{},
			matchSubresources:   []string{"/"},
			want:                true,
		},
		{
			name:                "subresource does not match root",
			requestSubresources: []string{"x"},
			matchSubresources:   []string{"/"},
			want:                false,
		},
		{
			name:                "root does not match subresource",
			requestSubresources: []string{},
			matchSubresources:   []string{"/x"},
			want:                false,
		},
		{
			name:                "root matches root and subresource",
			requestSubresources: []string{},
			matchSubresources:   []string{"/", "/x"},
			want:                true,
		},
		{
			name:                "subresource matches root and subresource",
			requestSubresources: []string{"x"},
			matchSubresources:   []string{"/", "/x"},
			want:                true,
		},
		{
			name:                "subresource matches multiple",
			requestSubresources: []string{"y"},
			matchSubresources:   []string{"/x", "/y"},
			want:                true,
		},
		{
			name:                "nested subresource match",
			requestSubresources: []string{"x", "y", "z"},
			matchSubresources:   []string{"/x/y/z"},
			want:                true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := Request{Subresources: tt.requestSubresources}
			if got := r.SubresourceIn(tt.matchSubresources); got != tt.want {
				t.Errorf("Request.SubresourceIn() = %v, want %v", got, tt.want)
			}
		})
	}
}
