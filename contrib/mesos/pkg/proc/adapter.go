/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package proc

import (
	"fmt"
)

type processAdapter struct {
	Process
	delegate Doer
}

// reportAnyError waits for an error to arrive from source, or for the process to end;
// errors are reported through errOnce. returns true if an error is reported through
// errOnce, otherwise false.
func (p *processAdapter) reportAnyError(source <-chan error, errOnce ErrorOnce) bool {
	select {
	case err, ok := <-source:
		if ok && err != nil {
			// failed to schedule/execute the action
			errOnce.Report(err)
			return true
		}
		// action was scheduled/executed just fine.
	case <-p.Done():
		// double-check that there's no errror waiting for us in source
		select {
		case err, ok := <-source:
			if ok {
				// parent failed to schedule/execute the action
				errOnce.Report(err)
				return true
			}
		default:
		}
		errOnce.Report(errProcessTerminated)
		return true
	}
	return false
}

func (p *processAdapter) Do(a Action) <-chan error {
	errCh := NewErrorOnce(p.Done())
	go func() {
		ch := NewErrorOnce(p.Done())
		errOuter := p.Process.Do(func() {
			errInner := p.delegate.Do(a)
			ch.forward(errInner)
		})
		// order is important here: check errOuter before ch
		if p.reportAnyError(errOuter, errCh) {
			return
		}
		if !p.reportAnyError(ch.Err(), errCh) {
			errCh.Report(nil)
		}
	}()
	return errCh.Err()
}

// DoWith returns a process that, within its execution context, delegates to the specified Doer.
// Expect a panic if either the given Process or Doer are nil.
func DoWith(other Process, d Doer) Process {
	if other == nil {
		panic(fmt.Sprintf("cannot DoWith a nil process"))
	}
	if d == nil {
		panic(fmt.Sprintf("cannot DoWith a nil doer"))
	}
	return &processAdapter{
		Process:  other,
		delegate: d,
	}
}
