// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package windriver

import (
	"golang.org/x/exp/shiny/driver/internal/errscreen"
	"golang.org/x/exp/shiny/driver/internal/win32"
	"golang.org/x/exp/shiny/screen"
)

// Main is called by the program's main function to run the graphical
// application.
//
// It calls f on the Screen, possibly in a separate goroutine, as some OS-
// specific libraries require being on 'the main thread'. It returns when f
// returns.
func Main(f func(screen.Screen)) {
	if err := win32.Main(func() { f(theScreen) }); err != nil {
		f(errscreen.Stub(err))
	}
}
