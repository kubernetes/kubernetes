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

package main

import (
	"go/parser"
	"go/token"
	"testing"
)

func TestCheckFile(t *testing.T) {
	tests := []struct {
		name string
		src  string
		want int
	}{
		{
			name: "skip in a closure in the test body",
			src: `var _ = SIGDescribe("x", func() {
	framework.ConformanceIt("a", func(ctx context.Context) {
		wait(func() { e2eskipper.SkipUnlessProviderIs("gce") })
	})
})`,
			want: 1,
		},
		{
			name: "skip in BeforeEach of an enclosing container",
			src: `var _ = SIGDescribe("x", func() {
	ginkgo.BeforeEach(func() { ginkgo.Skip("no") })
	ginkgo.Context("y", func() {
		framework.ConformanceIt("a", func(ctx context.Context) {})
		framework.ConformanceIt("b", func(ctx context.Context) {})
	})
})`,
			want: 2,
		},
		{
			name: "skips that do not reach a conformance test",
			src: `var _ = SIGDescribe("x", func() {
	ginkgo.Context("y", func() {
		ginkgo.BeforeEach(func() { e2eskipper.SkipUnlessProviderIs("gce") })
		ginkgo.It("a", func(ctx context.Context) { e2eskipper.Skipf("no") })
	})
	framework.ConformanceIt("b", func(ctx context.Context) {})
})`,
		},
		{
			name: "skip on Windows nodes is allowed",
			src: `var _ = SIGDescribe("x", func() {
	ginkgo.BeforeEach(func() { e2eskipper.SkipIfNodeOSDistroIs("windows") })
	framework.ConformanceIt("a", func(ctx context.Context) {})
})`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, "x.go", "package x\n"+tt.src, 0)
			if err != nil {
				t.Fatal(err)
			}
			if got := checkFile(fset, file); len(got) != tt.want {
				t.Errorf("got %d problems, want %d: %v", len(got), tt.want, got)
			}
		})
	}
}
