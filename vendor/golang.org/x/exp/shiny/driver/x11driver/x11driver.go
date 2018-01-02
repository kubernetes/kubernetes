// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package x11driver provides the X11 driver for accessing a screen.
package x11driver // import "golang.org/x/exp/shiny/driver/x11driver"

// TODO: figure out what to say about the responsibility for users of this
// package to check any implicit dependencies' LICENSEs. For example, the
// driver might use third party software outside of golang.org/x, like an X11
// or OpenGL library.

import (
	"fmt"

	"github.com/BurntSushi/xgb"
	"github.com/BurntSushi/xgb/render"
	"github.com/BurntSushi/xgb/shm"

	"golang.org/x/exp/shiny/driver/internal/errscreen"
	"golang.org/x/exp/shiny/screen"
)

// Main is called by the program's main function to run the graphical
// application.
//
// It calls f on the Screen, possibly in a separate goroutine, as some OS-
// specific libraries require being on 'the main thread'. It returns when f
// returns.
func Main(f func(screen.Screen)) {
	if err := main(f); err != nil {
		f(errscreen.Stub(err))
	}
}

func main(f func(screen.Screen)) (retErr error) {
	xc, err := xgb.NewConn()
	if err != nil {
		return fmt.Errorf("x11driver: xgb.NewConn failed: %v", err)
	}
	defer func() {
		if retErr != nil {
			xc.Close()
		}
	}()

	if err := render.Init(xc); err != nil {
		return fmt.Errorf("x11driver: render.Init failed: %v", err)
	}
	if err := shm.Init(xc); err != nil {
		return fmt.Errorf("x11driver: shm.Init failed: %v", err)
	}

	s, err := newScreenImpl(xc)
	if err != nil {
		return err
	}
	f(s)
	// TODO: tear down the s.run goroutine? It's probably not worth the
	// complexity of doing it cleanly, if the app is about to exit anyway.
	return nil
}
