// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensupport

import (
	"net/http"
	"testing"
)

func TestSendRequest(t *testing.T) {
	// Setting Accept-Encoding should give an error immediately.
	req, _ := http.NewRequest("GET", "url", nil)
	req.Header.Set("Accept-Encoding", "")
	_, err := SendRequest(nil, nil, req)
	if err == nil {
		t.Error("got nil, want error")
	}
}
