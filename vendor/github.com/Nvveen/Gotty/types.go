// Copyright 2012 Neal van Veen. All rights reserved.
// Usage of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package gotty

type TermInfo struct {
	boolAttributes map[string]bool
	numAttributes  map[string]int16
	strAttributes  map[string]string
	// The various names of the TermInfo file.
	Names []string
}

type stacker interface {
}
type stack []stacker

type parser struct {
	st         stack
	parameters []stacker
	dynamicVar map[byte]stacker
}
