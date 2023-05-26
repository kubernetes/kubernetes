// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.9
// +build !go1.9

package catalog

import "golang.org/x/text/internal/catmsg"

// A Message holds a collection of translations for the same phrase that may
// vary based on the values of substitution arguments.
type Message interface {
	catmsg.Message
}

func firstInSequence(m []Message) catmsg.Message {
	a := []catmsg.Message{}
	for _, m := range m {
		a = append(a, m)
	}
	return catmsg.FirstOf(a)
}
