/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// Package addargs lets you pass arguments through opaque processes like json
// marshalling.
package addargs

import (
	"fmt"
	"regexp"
	"runtime"
	"sync"
	"sync/atomic"
)

// Invoke calls 'f' with the given arg as thread local storage. 'f', or
// anything called by 'f', can call Get() to get arg. The purpose of this is to
// pass arguments through systems that don't allow suppling contexts (e.g., the
// standard json package).
//
// Note: If you nest calls to Invoke, each will shadow the prior ones.
func Invoke(arg interface{}, f func() error) error {
	return addAndCall(atomic.AddUint64(&nextUniqueKey, 1), arg, f)
}

// We do this in two steps. Can't rely on arg's memory location to be unique,
// or take its address and rely on that-- addresses to things on the stack can
// magically change when go allocates more memory for your stack. The tests
// catch this.
func addAndCall(key uint64, arg interface{}, f func() error) error {
	tlsKey, _ := computeKey(true)
	setArg(tlsKey, arg)
	defer setArg(tlsKey, nil)
	return f()
}

// Gets (if present) the given arg from thread local storage.
func Get() (interface{}, bool) {
	tlsKey, ok := computeKey(false)
	if !ok {
		return nil, false
	}

	lock.Lock()
	defer lock.Unlock()
	arg, ok := allArgs[tlsKey]
	return arg, ok
}

func computeKey(mustSucceed bool) (string, bool) {
	stack := stackPool.Get().([]byte)
	defer stackPool.Put(stack)
	runtime.Stack(stack, false)
	key, ok := tlsKey(stack)
	if !ok && mustSucceed {
		panic(fmt.Sprintf("Could not find addargs.addtls in: \n\n%s\n\n", stack))
	}
	return key, ok
}

// tlsKey returns the tlsKey for this
func tlsKey(stack []byte) (string, bool) {
	matches := keyRE.FindAllSubmatch(stack, 1)
	if len(matches) < 1 {
		return "", false
	}
	match := matches[0]
	if len(match) < 2 {
		return "", false
	}
	return string(match[1]), true
}

var (
	// match lines like: `k8s.io/kubernetes/pkg/util/addargs.addAndCall(0xc8200165d0, 0xc8200b1f58, 0x0, 0x0)`
	// Relies on the first argument being globally unique.
	// The tests will catch this if it is wrong.
	keyRE = regexp.MustCompile(`addargs.addAndCall\((0x[0-9a-f]{1,16}),`)

	allArgs       = map[string]interface{}{}
	lock          sync.Mutex
	stackPool     = sync.Pool{New: func() interface{} { return make([]byte, 50000) }}
	nextUniqueKey = uint64(0x111000000)
)

func setArg(key string, arg interface{}) {
	lock.Lock()
	defer lock.Unlock()
	allArgs[key] = arg
}
