/*
Copyright 2014 The Kubernetes Authors.

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

package diff

import (
	"bytes"
	"fmt"
	"strings"
	"text/tabwriter"

	"k8s.io/apimachinery/pkg/util/dump"
)

// ObjectGoPrintSideBySide prints a and b as textual dumps side by side,
// enabling easy visual scanning for mismatches.
func ObjectGoPrintSideBySide(a, b interface{}) string {
	sA := dump.Pretty(a)
	sB := dump.Pretty(b)

	linesA := strings.Split(sA, "\n")
	linesB := strings.Split(sB, "\n")
	width := 0
	for _, s := range linesA {
		l := len(s)
		if l > width {
			width = l
		}
	}
	for _, s := range linesB {
		l := len(s)
		if l > width {
			width = l
		}
	}
	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, width, 0, 1, ' ', 0)
	max := len(linesA)
	if len(linesB) > max {
		max = len(linesB)
	}
	for i := 0; i < max; i++ {
		var a, b string
		if i < len(linesA) {
			a = linesA[i]
		}
		if i < len(linesB) {
			b = linesB[i]
		}
		_, _ = fmt.Fprintf(w, "%s\t%s\n", a, b)
	}
	_ = w.Flush()
	return buf.String()
}
