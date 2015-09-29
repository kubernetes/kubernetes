// Copyright 2012 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mux

import (
	"net/http"
	"testing"
)

func BenchmarkMux(b *testing.B) {
	router := new(Router)
	handler := func(w http.ResponseWriter, r *http.Request) {}
	router.HandleFunc("/v1/{v1}", handler)

	request, _ := http.NewRequest("GET", "/v1/anything", nil)
	for i := 0; i < b.N; i++ {
		router.ServeHTTP(nil, request)
	}
}
