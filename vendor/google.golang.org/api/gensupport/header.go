// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"fmt"
	"runtime"
	"strings"
)

// GoogleClientHeader returns the value to use for the x-goog-api-client
// header, which is used internally by Google.
func GoogleClientHeader(generatorVersion, clientElement string) string {
	elts := []string{"gl-go/" + strings.Replace(runtime.Version(), " ", "_", -1)}
	if clientElement != "" {
		elts = append(elts, clientElement)
	}
	elts = append(elts, fmt.Sprintf("gdcl/%s", generatorVersion))
	return strings.Join(elts, " ")
}
