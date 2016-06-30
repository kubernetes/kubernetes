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

package tasks

type Events interface {
	// Close stops delivery of events in the completion and errors channels; callers must close this when they intend to no longer read from completion() or errors()
	Close() Events

	// Completion reports Completion events as they happen
	Completion() <-chan *Completion

	// Done returns a signal chan that closes when all tasks have completed and there are no more events to deliver
	Done() <-chan struct{}
}

type eventsImpl struct {
	tc             chan *Completion
	stopForwarding chan struct{}
	done           <-chan struct{}
}

func newEventsImpl(tcin <-chan *Completion, done <-chan struct{}) *eventsImpl {
	ei := &eventsImpl{
		tc:             make(chan *Completion),
		stopForwarding: make(chan struct{}),
		done:           done,
	}
	go func() {
		defer close(ei.tc)
		forwardCompletionUntil(tcin, ei.tc, ei.stopForwarding, done, nil)
	}()
	return ei
}

func (e *eventsImpl) Close() Events                  { close(e.stopForwarding); return e }
func (e *eventsImpl) Completion() <-chan *Completion { return e.tc }
func (e *eventsImpl) Done() <-chan struct{}          { return e.done }

// forwardCompletionUntil is a generic pipe that forwards objects between channels.
// if discard is closed, objects are silently dropped.
// if tap != nil then it's invoked for each object as it's read from tin, but before it's written to tch.
// returns when either reading from tin completes (no more objects, and is closed), or else
// abort is closed, which ever happens first.
func forwardCompletionUntil(tin <-chan *Completion, tch chan<- *Completion, discard <-chan struct{}, abort <-chan struct{}, tap func(*Completion, bool)) {
	var tc *Completion
	var ok bool
forwardLoop:
	for {
		select {
		case tc, ok = <-tin:
			if !ok {
				return
			}
			if tap != nil {
				tap(tc, false)
			}
			select {
			case <-abort:
				break forwardLoop
			case <-discard:
			case tch <- tc:
			}
		case <-abort:
			// best effort
			select {
			case tc, ok = <-tin:
				if ok {
					if tap != nil {
						tap(tc, true)
					}
					break forwardLoop
				}
			default:
			}
			return
		}
	}
	// best effort
	select {
	case tch <- tc:
	case <-discard:
	default:
	}
}
