// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package internal contains support packages for oauth2 package.
package internal

import (
	"fmt"
	"testing"
)

func TestRegisterBrokenAuthHeaderProvider(t *testing.T) {
	RegisterBrokenAuthHeaderProvider("https://aaa.com/")
	tokenURL := "https://aaa.com/token"
	if providerAuthHeaderWorks(tokenURL) {
		t.Errorf("got %q as unbroken; want broken", tokenURL)
	}
}

func Test_providerAuthHeaderWorks(t *testing.T) {
	for _, p := range brokenAuthHeaderProviders {
		if providerAuthHeaderWorks(p) {
			t.Errorf("got %q as unbroken; want broken", p)
		}
		p := fmt.Sprintf("%ssomesuffix", p)
		if providerAuthHeaderWorks(p) {
			t.Errorf("got %q as unbroken; want broken", p)
		}
	}
	p := "https://api.not-in-the-list-example.com/"
	if !providerAuthHeaderWorks(p) {
		t.Errorf("got %q as unbroken; want broken", p)
	}
}
