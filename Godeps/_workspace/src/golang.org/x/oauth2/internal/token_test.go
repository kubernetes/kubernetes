// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package internal contains support packages for oauth2 package.
package internal

import (
	"fmt"
	"testing"
)

func Test_providerAuthHeaderWorks(t *testing.T) {
	for _, p := range brokenAuthHeaderProviders {
		if providerAuthHeaderWorks(p) {
			t.Errorf("URL: %s not found in list", p)
		}
		p := fmt.Sprintf("%ssomesuffix", p)
		if providerAuthHeaderWorks(p) {
			t.Errorf("URL: %s not found in list", p)
		}
	}
	p := "https://api.not-in-the-list-example.com/"
	if !providerAuthHeaderWorks(p) {
		t.Errorf("URL: %s found in list", p)
	}

}
