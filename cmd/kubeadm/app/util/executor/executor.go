/*
Copyright 2017 The Kubernetes Authors.

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

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"sort"
	"sync"
)

func New(out io.Writer, units []*Unit) (*Queue, error) {
	q := &Queue{
		units: map[string]*Unit{},
		waits: map[string]chan struct{}{},
		queue: []string{},
		out:   out,
	}

	for _, u := range units {
		q.units[u.Name] = u
		q.waits[u.Name] = make(chan struct{})
	}

	if err := q.validate(); err != nil {
		return nil, err
	}

	if err := q.sort(); err != nil {
		return nil, err
	}

	return q, nil
}

type Queue struct {
	units map[string]*Unit
	waits map[string]chan struct{}
	queue []string

	mu  sync.Mutex
	idx int

	outMu sync.Mutex
	out   io.Writer
}

func (q *Queue) Run(ctx context.Context, workers int) error {
	ctx, cancel := context.WithCancel(ctx)

	var (
		wg       sync.WaitGroup
		errGuard sync.Mutex
		err      error
	)

	done := make(chan struct{})

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if workErr := q.work(ctx); workErr != nil {
				errGuard.Lock()
				err = workErr
				errGuard.Unlock()

				done <- struct{}{}
			}
		}()
	}

	go func() {
		wg.Wait()
		done <- struct{}{}
	}()

	<-done
	cancel()
	return err
}

func (q *Queue) work(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return nil
		default:
			q.mu.Lock()
			idx := q.idx
			q.idx++
			q.mu.Unlock()

			if idx >= len(q.queue) {
				return nil
			}
			if err := q.doOne(ctx, q.queue[idx]); err != nil {
				return err
			}
		}
	}
}

func (q *Queue) doOne(ctx context.Context, name string) error {
	u := q.units[name]
	for _, dep := range u.Deps {
		select {
		case <-ctx.Done():
			return nil
		case _, ok := <-q.waits[dep]:
			if ok {
				return fmt.Errorf("unexpected channel open")
			}
		}
	}

	var out bytes.Buffer

	err := u.Action(ctx, &out)
	if err != nil {
		return err
	}

	q.outMu.Lock()
	out.WriteTo(q.out)
	q.outMu.Unlock()

	close(q.waits[name])

	return nil
}

func (q *Queue) sort() error {
	sorted, err := topoSort(q.units)
	if err != nil {
		return err
	}
	q.queue = sorted
	return nil
}

// topoSort implements Kahn's algorithm to topologically sort the queue by
// dependency. This sort is not stable. This sort returns an error if it
// detects a circular dependency.
func topoSort(units map[string]*Unit) ([]string, error) {
	type node struct {
		name string
		in   []string
		out  []string
	}

	nodes := map[string]*node{}
	for _, u := range units {
		nodes[u.Name] = &node{
			name: u.Name,
			in:   u.Deps,
		}
	}
	for _, u := range units {
		for _, dep := range u.Deps {
			nodes[dep].out = append(nodes[dep].out, u.Name)
		}
	}

	for _, n := range nodes {
		sort.Strings(n.in)
		sort.Strings(n.out)
	}

	sorted := []string{}

	roots := []string{}
	for _, n := range nodes {
		if len(n.in) == 0 {
			roots = append(roots, n.name)
		}
	}
	sort.Strings(roots)

	for len(roots) != 0 {
		cur := nodes[roots[0]]
		sorted = append(sorted, cur.name)

		for _, out := range cur.out {
			found := false
			n := nodes[out]
			for i, in := range n.in {
				if in != cur.name {
					continue
				}
				n.in = append(n.in[:i], n.in[i+1:]...)
				found = true
				break
			}
			if !found {
				panic("programmer error")
			}
			if len(n.in) == 0 {
				roots = append(roots, n.name)
			}
		}
		roots = roots[1:]
	}

	if len(sorted) != len(nodes) {
		return nil, fmt.Errorf("circular dependency")
	}

	return sorted, nil
}

func (q *Queue) validate() error {
	for _, u := range q.units {
		for _, dep := range u.Deps {
			if _, ok := q.units[dep]; !ok {
				return fmt.Errorf("unknown dep: %v", dep)
			}
		}
	}
	return nil
}
