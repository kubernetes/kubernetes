// Package gc experiments with providing central gc tooling to ensure
// deterministic resource removal within containerd.
//
// For now, we just have a single exported implementation that can be used
// under certain use cases.
package gc

import (
	"context"
	"sync"
)

// ResourceType represents type of resource at a node
type ResourceType uint8

// Node presents a resource which has a type and key,
// this node can be used to lookup other nodes.
type Node struct {
	Type      ResourceType
	Namespace string
	Key       string
}

// Tricolor implements basic, single-thread tri-color GC. Given the roots, the
// complete set and a refs function, this function returns a map of all
// reachable objects.
//
// Correct usage requires that the caller not allow the arguments to change
// until the result is used to delete objects in the system.
//
// It will allocate memory proportional to the size of the reachable set.
//
// We can probably use this to inform a design for incremental GC by injecting
// callbacks to the set modification algorithms.
func Tricolor(roots []Node, refs func(ref Node) ([]Node, error)) (map[Node]struct{}, error) {
	var (
		grays     []Node                // maintain a gray "stack"
		seen      = map[Node]struct{}{} // or not "white", basically "seen"
		reachable = map[Node]struct{}{} // or "block", in tri-color parlance
	)

	grays = append(grays, roots...)

	for len(grays) > 0 {
		// Pick any gray object
		id := grays[len(grays)-1] // effectively "depth first" because first element
		grays = grays[:len(grays)-1]
		seen[id] = struct{}{} // post-mark this as not-white
		rs, err := refs(id)
		if err != nil {
			return nil, err
		}

		// mark all the referenced objects as gray
		for _, target := range rs {
			if _, ok := seen[target]; !ok {
				grays = append(grays, target)
			}
		}

		// mark as black when done
		reachable[id] = struct{}{}
	}

	return reachable, nil
}

// ConcurrentMark implements simple, concurrent GC. All the roots are scanned
// and the complete set of references is formed by calling the refs function
// for each seen object. This function returns a map of all object reachable
// from a root.
//
// Correct usage requires that the caller not allow the arguments to change
// until the result is used to delete objects in the system.
//
// It will allocate memory proportional to the size of the reachable set.
func ConcurrentMark(ctx context.Context, root <-chan Node, refs func(context.Context, Node, func(Node)) error) (map[Node]struct{}, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		grays = make(chan Node)
		seen  = map[Node]struct{}{} // or not "white", basically "seen"
		wg    sync.WaitGroup

		errOnce sync.Once
		refErr  error
	)

	go func() {
		for gray := range grays {
			if _, ok := seen[gray]; ok {
				wg.Done()
				continue
			}
			seen[gray] = struct{}{} // post-mark this as non-white

			go func(gray Node) {
				defer wg.Done()

				send := func(n Node) {
					wg.Add(1)
					select {
					case grays <- n:
					case <-ctx.Done():
						wg.Done()
					}
				}

				if err := refs(ctx, gray, send); err != nil {
					errOnce.Do(func() {
						refErr = err
						cancel()
					})
				}

			}(gray)
		}
	}()

	for r := range root {
		wg.Add(1)
		select {
		case grays <- r:
		case <-ctx.Done():
			wg.Done()
		}

	}

	// Wait for outstanding grays to be processed
	wg.Wait()

	close(grays)

	if refErr != nil {
		return nil, refErr
	}
	if cErr := ctx.Err(); cErr != nil {
		return nil, cErr
	}

	return seen, nil
}

// Sweep removes all nodes returned through the channel which are not in
// the reachable set by calling the provided remove function.
func Sweep(reachable map[Node]struct{}, all []Node, remove func(Node) error) error {
	// All black objects are now reachable, and all white objects are
	// unreachable. Free those that are white!
	for _, node := range all {
		if _, ok := reachable[node]; !ok {
			if err := remove(node); err != nil {
				return err
			}
		}
	}

	return nil
}
