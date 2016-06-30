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
//   op := StartPortAllocationOperation(...)
//   defer op.Finish
//   ...
//   write(updatedOwner)
///  op.Commit()
type portAllocationOperation struct {
	pa              Interface
	allocated       []int
	releaseDeferred []int
	shouldRollback  bool
}

// Creates a portAllocationOperation, tracking a set of allocations & releases
func StartOperation(pa Interface) *portAllocationOperation {
	op := &portAllocationOperation{}
	op.pa = pa
	op.allocated = []int{}
	op.releaseDeferred = []int{}
	op.shouldRollback = true
	return op
}

// Will rollback unless marked as shouldRollback = false by a Commit().  Call from a defer block
func (op *portAllocationOperation) Finish() {
	if op.shouldRollback {
		op.Rollback()
	}
}

// (Try to) undo any operations we did
func (op *portAllocationOperation) Rollback() []error {
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
func (op *portAllocationOperation) Commit() []error {
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
func (op *portAllocationOperation) Allocate(port int) error {
	err := op.pa.Allocate(port)
	if err == nil {
		op.allocated = append(op.allocated, port)
	}
	return err
}

// Allocates a port, and record it for future rollback
func (op *portAllocationOperation) AllocateNext() (int, error) {
	port, err := op.pa.AllocateNext()
	if err == nil {
		op.allocated = append(op.allocated, port)
	}
	return port, err
}

// Marks a port so that it will be released if this operation Commits
func (op *portAllocationOperation) ReleaseDeferred(port int) {
	op.releaseDeferred = append(op.releaseDeferred, port)
}
