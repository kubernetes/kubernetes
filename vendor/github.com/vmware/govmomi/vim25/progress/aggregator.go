/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package progress

import "sync"

type Aggregator struct {
	downstream Sinker
	upstream   chan (<-chan Report)

	done chan struct{}
	w    sync.WaitGroup
}

func NewAggregator(s Sinker) *Aggregator {
	a := &Aggregator{
		downstream: s,
		upstream:   make(chan (<-chan Report)),

		done: make(chan struct{}),
	}

	a.w.Add(1)
	go a.loop()

	return a
}

func (a *Aggregator) loop() {
	defer a.w.Done()

	dch := a.downstream.Sink()
	defer close(dch)

	for {
		select {
		case uch := <-a.upstream:
			// Drain upstream channel
			for e := range uch {
				dch <- e
			}
		case <-a.done:
			return
		}
	}
}

func (a *Aggregator) Sink() chan<- Report {
	ch := make(chan Report)
	a.upstream <- ch
	return ch
}

// Done marks the aggregator as done. No more calls to Sink() may be made and
// the downstream progress report channel will be closed when Done() returns.
func (a *Aggregator) Done() {
	close(a.done)
	a.w.Wait()
}
