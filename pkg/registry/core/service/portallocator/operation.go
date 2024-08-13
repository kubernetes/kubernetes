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

package portallocator

// Encapsulates the semantics of a port allocation 'transaction':
// It is better to leak ports than to double-allocate them,
// so we allocate immediately, but defer release.
// On commit we best-effort release the deferred releases.
// On rollback we best-effort release any allocations we did.
//
// Pattern for use:
//
//	op := StartPortAllocationOperation(...)
//	defer op.Finish
//	...
//	write(updatedOwner)
//
// /  op.Commit()
type PortAllocationOperation struct {
	pa              Interface
	allocated       []int
	releaseDeferred []int
	shouldRollback  bool
	dryRun          bool
}

// Creates a portAllocationOperation, tracking a set of allocations & releases
// If dryRun is specified, never actually allocate or release anything
func StartOperation(pa Interface, dryRun bool) *PortAllocationOperation {
	op := &PortAllocationOperation{}
	op.pa = pa
	op.allocated = []int{}
	op.releaseDeferred = []int{}
	op.shouldRollback = true
	op.dryRun = dryRun
	return op
}

// Will rollback unless marked as shouldRollback = false by a Commit().  Call from a defer block
func (op *PortAllocationOperation) Finish() {
	if op.shouldRollback {
		op.Rollback()
	}
}

// (Try to) undo any operations we did
func (op *PortAllocationOperation) Rollback() []error {
	if op.dryRun {
		return nil
	}

	errors := []error{}

	for _, allocated := range op.allocated {
		err := op.pa.Release(allocated)
		if err != nil {
			errors = append(errors, err)
		}
	}

	if len(errors) == 0 {
		return nil
	}
	return errors
}

// (Try to) perform any deferred operations.
// Note that even if this fails, we don't rollback; we always want to err on the side of over-allocation,
// and Commit should be called _after_ the owner is written
func (op *PortAllocationOperation) Commit() []error {
	if op.dryRun {
		return nil
	}

	errors := []error{}

	for _, release := range op.releaseDeferred {
		err := op.pa.Release(release)
		if err != nil {
			errors = append(errors, err)
		}
	}

	// Even on error, we don't rollback
	// Problems should be fixed by an eventual reconciliation / restart
	op.shouldRollback = false

	if len(errors) == 0 {
		return nil
	}

	return errors
}

// Allocates a port, and record it for future rollback
func (op *PortAllocationOperation) Allocate(port int) error {
	if op.dryRun {
		if op.pa.Has(port) {
			return ErrAllocated
		}
		for _, a := range op.allocated {
			if port == a {
				return ErrAllocated
			}
		}
		op.allocated = append(op.allocated, port)
		return nil
	}

	err := op.pa.Allocate(port)
	if err == nil {
		op.allocated = append(op.allocated, port)
	}
	return err
}

// Allocates a port, and record it for future rollback
func (op *PortAllocationOperation) AllocateNext() (int, error) {
	if op.dryRun {
		// Find the max element of the allocated ports array.
		// If no ports are already being allocated by this operation,
		// then choose a sensible guess for a dummy port number
		var lastPort int
		for _, allocatedPort := range op.allocated {
			if allocatedPort > lastPort {
				lastPort = allocatedPort
			}
		}
		if len(op.allocated) == 0 {
			lastPort = 32768
		}

		// Try to find the next non allocated port.
		// If too many ports are full, just reuse one,
		// since this is just a dummy value.
		for port := lastPort + 1; port < 100; port++ {
			err := op.Allocate(port)
			if err == nil {
				return port, nil
			}
		}
		op.allocated = append(op.allocated, lastPort+1)
		return lastPort + 1, nil
	}

	port, err := op.pa.AllocateNext()
	if err == nil {
		op.allocated = append(op.allocated, port)
	}
	return port, err
}

// Marks a port so that it will be released if this operation Commits
func (op *PortAllocationOperation) ReleaseDeferred(port int) {
	op.releaseDeferred = append(op.releaseDeferred, port)
}
