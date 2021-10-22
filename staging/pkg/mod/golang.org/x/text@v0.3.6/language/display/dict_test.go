// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package display

import (
	"fmt"
	"testing"

	"golang.org/x/text/internal/testtext"
)

func TestLinking(t *testing.T) {
	base := getSize(t, `display.Tags(language.English).Name(language.English)`)
	compact := getSize(t, `display.English.Languages().Name(language.English)`)

	if d := base - compact; d < 1.5*1024*1024 {
		t.Errorf("size(base) - size(compact) = %d - %d = was %d; want > 1.5MB", base, compact, d)
	}
}

func getSize(t *testing.T, main string) int {
	size, err := testtext.CodeSize(fmt.Sprintf(body, main))
	if err != nil {
		t.Skipf("skipping link size test; binary size could not be determined: %v", err)
	}
	return size
}

const body = `package main
import (
	"golang.org/x/text/language"
	"golang.org/x/text/language/display"
)
func main() {
	%s
}
`
