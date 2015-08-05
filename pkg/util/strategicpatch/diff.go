/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package strategicpatch

import (
	"io"
	"strings"
)

type diffWriter struct {
	io.Writer
	Depth int
}

func (dw *diffWriter) WriteLine(prefix, out string) {
	out = prefix + strings.Repeat("  ", dw.Depth) + out + "\n"
	dw.Write([]byte(out))
}

func (dw *diffWriter) WriteSame(out string) {
	dw.WriteLine(" ", out)
}

func (dw *diffWriter) WriteAddition(out string) {
	dw.WriteLine("+", out)
}

func (dw *diffWriter) WriteRemoval(out string) {
	dw.WriteLine("-", out)
}

func (dw *diffWriter) DepthInc() {
	dw.Depth++
}

func (dw *diffWriter) DepthDec() {
	dw.Depth--
}
