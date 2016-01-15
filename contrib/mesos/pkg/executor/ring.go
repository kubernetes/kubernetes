/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package executor

type ring struct {
	input          <-chan *RegisteredPod
	output         chan *RegisteredPod
	handleOverflow func(*RegisteredPod)
}

func newRing(in <-chan *RegisteredPod, out chan *RegisteredPod, dropOverflow bool) *ring {
	r := &ring{
		input:  in,
		output: out,
	}
	if dropOverflow {
		r.handleOverflow = func(x *RegisteredPod) {
			select {
			case <-r.output: // drop the oldest item
			default: // someone already took everything?!
			}
			r.output <- x
		}
	} else {
		r.handleOverflow = func(x *RegisteredPod) {
			// don't drop anything, block until it writes
			r.output <- x
		}
	}
	return r
}

func (r *ring) run() {
	defer close(r.output)
	for x := range r.input {
		select {
		case r.output <- x:
		default:
			r.handleOverflow(x)
		}
	}
}
