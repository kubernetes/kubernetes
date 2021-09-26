// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:generate gotext -srclang=en update -out=catalog_gen.go -lang=en,zh

import (
	"net/http"

	"golang.org/x/text/cmd/gotext/examples/extract_http/pkg"
)

func main() {
	http.Handle("/generize", http.HandlerFunc(pkg.Generize))
}
