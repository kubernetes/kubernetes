// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pflag

import (
	"io/ioutil"
	"os"
)

// Additional routines compiled into the package only during testing.

// ResetForTesting clears all flag state and sets the usage function as directed.
// After calling ResetForTesting, parse errors in flag handling will not
// exit the program.
func ResetForTesting(usage func()) {
	CommandLine = &FlagSet{
		name:          os.Args[0],
		errorHandling: ContinueOnError,
		output:        ioutil.Discard,
	}
	Usage = usage
}

// GetCommandLine returns the default FlagSet.
func GetCommandLine() *FlagSet {
	return CommandLine
}
