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

package runtime

import (
	"os"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/util/runtime"
)

type Signal <-chan struct{}

// return a func that will close the signal chan.
// multiple invocations of the returned func will not generate a panic.
// two funcs from separate invocations of Closer() (on the same sig chan) will cause a panic if both invoked.
// for example:
//     // good
//     x := runtime.After(func() { ... })
//     f := x.Closer()
//     f()
//     f()
//
//     // bad
//     x := runtime.After(func() { ... })
//     f := x.Closer()
//     g := x.Closer()
//     f()
//     g() // this will panic
func Closer(sig chan<- struct{}) func() {
	var once sync.Once
	return func() {
		once.Do(func() { close(sig) })
	}
}

// upon receiving signal sig invoke function f and immediately return a signal
// that indicates f's completion. used to chain handler funcs, for example:
//    On(job.Done(), response.Send).Then(wg.Done)
func (sig Signal) Then(f func()) Signal {
	if sig == nil {
		return nil
	}
	return On(sig, f)
}

// execute a callback function after the specified signal chan closes.
// immediately returns a signal that indicates f's completion.
func On(sig <-chan struct{}, f func()) Signal {
	if sig == nil {
		return nil
	}
	return After(func() {
		<-sig
		if f != nil {
			f()
		}
	})
}

func OnOSSignal(sig <-chan os.Signal, f func(os.Signal)) Signal {
	if sig == nil {
		return nil
	}
	return After(func() {
		if s, ok := <-sig; ok && f != nil {
			f(s)
		}
	})
}

// spawn a goroutine to execute a func, immediately returns a chan that closes
// upon completion of the func. returns a nil signal chan if the given func is nil.
func After(f func()) Signal {
	ch := make(chan struct{})
	go func() {
		defer close(ch)
		defer runtime.HandleCrash()
		if f != nil {
			f()
		}
	}()
	return Signal(ch)
}

// periodically execute the given function, stopping once stopCh is closed.
// this func blocks until stopCh is closed, it's intended to be run as a goroutine.
func Until(f func(), period time.Duration, stopCh <-chan struct{}) {
	if f == nil {
		return
	}
	for {
		select {
		case <-stopCh:
			return
		default:
		}
		func() {
			defer runtime.HandleCrash()
			f()
		}()
		select {
		case <-stopCh:
		case <-time.After(period):
		}
	}
}
