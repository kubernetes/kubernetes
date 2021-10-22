// Copyright 2014 Google LLC
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

package btree

import (
	"fmt"
	"io"
	"strings"
)

func (t *BTree) print(w io.Writer) {
	t.root.print(w, 0)
}

func (n *node) print(w io.Writer, level int) {
	indent := strings.Repeat("   ", level)
	if n == nil {
		fmt.Fprintf(w, "%s<nil>\n", indent)
		return
	}
	fmt.Fprintf(w, "%s%v\n", indent, n.items)
	for _, c := range n.children {
		c.print(w, level+1)
	}
}
