// Copyright (c) 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Package maxprocs lets Go programs easily configure runtime.GOMAXPROCS to
// match the configured Linux CPU quota. Unlike the top-level automaxprocs
// package, it lets the caller configure logging and handle errors.
package automaxprocs

import (
	"os"
	"runtime"
)

func init() {
	Set()
}

const _maxProcsKey = "GOMAXPROCS"

type config struct {
	procs          func(int, func(v float64) int) (int, CPUQuotaStatus, error)
	minGOMAXPROCS  int
	roundQuotaFunc func(v float64) int
}

// Set GOMAXPROCS to match the Linux container CPU quota (if any), returning
// any error encountered and an undo function.
//
// Set is a no-op on non-Linux systems and in Linux environments without a
// configured CPU quota.
func Set() error {
	cfg := &config{
		procs:          CPUQuotaToGOMAXPROCS,
		roundQuotaFunc: DefaultRoundFunc,
		minGOMAXPROCS:  1,
	}

	// Honor the GOMAXPROCS environment variable if present. Otherwise, amend
	// `runtime.GOMAXPROCS()` with the current process' CPU quota if the OS is
	// Linux, and guarantee a minimum value of 1. The minimum guaranteed value
	// can be overridden using `maxprocs.Min()`.
	if _, exists := os.LookupEnv(_maxProcsKey); exists {
		return nil
	}
	maxProcs, status, err := cfg.procs(cfg.minGOMAXPROCS, cfg.roundQuotaFunc)
	if err != nil {
		return err
	}
	if status == CPUQuotaUndefined {
		return nil
	}
	runtime.GOMAXPROCS(maxProcs)
	return nil
}
