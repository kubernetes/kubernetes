/*
Copyright 2016 The Kubernetes Authors.

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

/*
Package nestedpendingoperations is a modified implementation of
pkg/util/goroutinemap. It implements a data structure for managing go routines
by volume/pod name. It prevents the creation of new go routines if an existing
go routine for the volume already exists. It also allows multiple operations to
execute in parallel for the same volume as long as they are operating on
different pods.
*/
package nestedpendingoperations

import (
	"fmt"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	k8sRuntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

const (
	// EmptyUniquePodName is a UniquePodName for empty string.
	EmptyUniquePodName volumetypes.UniquePodName = volumetypes.UniquePodName("")

	// EmptyUniqueVolumeName is a UniqueVolumeName for empty string
	EmptyUniqueVolumeName v1.UniqueVolumeName = v1.UniqueVolumeName("")

	// EmptyNodeName is a NodeName for empty string
	EmptyNodeName types.NodeName = types.NodeName("")

	// EmptyOperationName for a placeholder
	EmptyOperationName = "placeholder"
)

// NestedPendingOperations defines the supported set of operations.
type NestedPendingOperations interface {

	// Run adds the concatenation of volumeName, podName, and nodeName to the list
	// of running operations and spawns a new go routine to run
	// generatedOperations.

	// volumeName, podName, and nodeName collectively form the operation key.
	// The following forms of operation keys are supported (two keys are designed
	// to be "matched" if we want to serialize their operations):
	// - volumeName empty, podName empty, nodeName exists
	//   This key matches the same nodeName, but empty volumeName and podName
	// - volumeName exists, podName empty, nodeName empty
	//   This key matches all other keys with the same volumeName.
	// - volumeName exists, podName exists, nodeName empty
	//   This key matches with:
	//   - the same volumeName and podName
	//   - the same volumeName, but empty podName
	// - volumeName exists, podName empty, nodeName exists
	//   This key matches with:
	//   - the same volumeName and nodeName
	//   - the same volumeName, but empty nodeName

	// If there is no operation with a matching key, the operation is allowed to
	// proceed.
	// If an operation with a matching key exists and the previous operation is
	// running, an AlreadyExists error is returned.
	// If an operation with a matching key exists and the previous operation
	// failed:
	// - If the previous operation has the same
	//   generatedOperations.operationName:
	//   - If the full exponential backoff period is satisfied, the operation is
	//     allowed to proceed.
	//   - Otherwise, an ExponentialBackoff error is returned.
	// - Otherwise, exponential backoff is reset and operation is allowed to
	//   proceed.

	// Once the operation is complete, the go routine is terminated. If the
	// operation succeeded, its corresponding key is removed from the list of
	// executing operations, allowing a new operation to be started with the key
	// without error. If it failed, the key remains and the exponential
	// backoff status is updated.
	Run(
		volumeName v1.UniqueVolumeName,
		podName volumetypes.UniquePodName,
		nodeName types.NodeName,
		generatedOperations volumetypes.GeneratedOperations) error

	// Wait blocks until all operations are completed. This is typically
	// necessary during tests - the test should wait until all operations finish
	// and evaluate results after that.
	Wait()

	// IsOperationPending returns true if an operation for the given volumeName
	// and one of podName or nodeName is pending, otherwise it returns false
	IsOperationPending(
		volumeName v1.UniqueVolumeName,
		podName volumetypes.UniquePodName,
		nodeName types.NodeName) bool

	// IsOperationSafeToRetry returns false if an operation for the given volumeName
	// and one of podName or nodeName is pending or in exponential backoff, otherwise it returns true
	IsOperationSafeToRetry(
		volumeName v1.UniqueVolumeName,
		podName volumetypes.UniquePodName,
		nodeName types.NodeName, operationName string) bool
}

// NewNestedPendingOperations returns a new instance of NestedPendingOperations.
func NewNestedPendingOperations(exponentialBackOffOnError bool) NestedPendingOperations {
	g := &nestedPendingOperations{
		operations: &operationsCache{
			pendingOps: make(map[v1.UniqueVolumeName]*volumeRelatedOperation),
			allOps:     make(map[operationKey]*operation),
		},
		exponentialBackOffOnError: exponentialBackOffOnError,
	}
	g.cond = sync.NewCond(&g.lock)
	return g
}

type nestedPendingOperations struct {
	operations                *operationsCache
	exponentialBackOffOnError bool
	cond                      *sync.Cond
	lock                      sync.RWMutex
}

type operationsCache struct {
	// recording pending operations, while the operation completes whenever success or not,
	// it will be always removed from here.
	pendingOps map[v1.UniqueVolumeName]*volumeRelatedOperation
	// stores operations with unique volumeName, podName, nodeName and operationName key
	allOps map[operationKey]*operation
}

type volumeRelatedOperation struct {
	// operationName for the volume without related pod or node
	operationName string
	// stores pending operations for the volume with related node
	nodeRelated map[types.NodeName]*nodeRelatedOperation
}

type nodeRelatedOperation struct {
	// operationName for the volume with related node
	operationName string
	// stores pending operations for the volume with related pod
	podRelated map[volumetypes.UniquePodName]*podRelatedOperation
}

type podRelatedOperation struct {
	// operationName for the volume with related node and pod
	operationName string
}

type operation struct {
	key              operationKey
	operationName    string
	operationPending bool
	expBackoff       exponentialbackoff.ExponentialBackoff
}

type operationKey struct {
	volumeName    v1.UniqueVolumeName
	podName       volumetypes.UniquePodName
	nodeName      types.NodeName
	operationName string
}

func (grm *nestedPendingOperations) Run(
	volumeName v1.UniqueVolumeName,
	podName volumetypes.UniquePodName,
	nodeName types.NodeName,
	generatedOperations volumetypes.GeneratedOperations) error {
	grm.lock.Lock()
	defer grm.lock.Unlock()

	opKey := operationKey{volumeName, podName, nodeName, generatedOperations.OperationName}

	if grm.operations.hasOperationPending(volumeName, podName, nodeName) {
		return NewAlreadyExistsError(opKey)
	}

	opExists, op := grm.operations.isOperationExists(opKey)
	if opExists {
		backOffErr := op.expBackoff.SafeToRetry(fmt.Sprintf("%+v", opKey))
		if backOffErr != nil {
			return backOffErr
		}
		// Update existing operation to mark as pending.
		grm.operations.updateOperation(opKey, true, false, nil)
	} else {
		// Create a new operation
		grm.operations.addOperation(&operation{
			key:              opKey,
			operationPending: true,
			operationName:    generatedOperations.OperationName,
			expBackoff:       exponentialbackoff.ExponentialBackoff{},
		})
	}

	go func() (eventErr, detailedErr error) {
		// Handle unhandled panics (very unlikely)
		defer k8sRuntime.HandleCrash()
		// Handle completion of and error, if any, from operationFunc()
		defer grm.operationComplete(opKey, &detailedErr)
		return generatedOperations.Run()
	}()

	return nil
}
func (grm *nestedPendingOperations) IsOperationSafeToRetry(
	volumeName v1.UniqueVolumeName,
	podName volumetypes.UniquePodName,
	nodeName types.NodeName,
	operationName string) bool {

	grm.lock.RLock()
	defer grm.lock.RUnlock()

	opKey := operationKey{volumeName, podName, nodeName, operationName}
	exists, op := grm.operations.isOperationExists(opKey)
	if !exists {
		return true
	}
	if op.operationPending {
		return false
	}
	backOffErr := op.expBackoff.SafeToRetry(fmt.Sprintf("%+v", opKey))
	if backOffErr != nil {
		return false
	}

	return true
}

func (grm *nestedPendingOperations) IsOperationPending(
	volumeName v1.UniqueVolumeName,
	podName volumetypes.UniquePodName,
	nodeName types.NodeName) bool {

	grm.lock.RLock()
	defer grm.lock.RUnlock()

	return grm.operations.hasOperationPending(volumeName, podName, nodeName)

}

func (grm *nestedPendingOperations) operationComplete(key operationKey, err *error) {
	// Defer operations are executed in Last-In is First-Out order. In this case
	// the lock is acquired first when operationCompletes begins, and is
	// released when the method finishes, after the lock is released cond is
	// signaled to wake waiting goroutine.
	defer grm.cond.Signal()
	grm.lock.Lock()
	defer grm.lock.Unlock()

	if *err == nil || !grm.exponentialBackOffOnError {
		// Operation completed without error, or exponentialBackOffOnError disabled
		grm.operations.deleteOperation(key)
		if *err != nil {
			// Log error
			klog.Errorf("operation %+v failed with: %v", key, *err)
		}
		return
	}

	// Operation completed with error and exponentialBackOffOnError Enabled
	existingOp, getOpErr := grm.operations.getOperation(key)
	if getOpErr != nil {
		// Failed to find existing operation
		klog.Errorf("Operation %+v completed. error: %v. exponentialBackOffOnError is enabled, but failed to get operation to update.",
			key,
			*err)
		// Clear pending state
		grm.operations.updateOperation(key, false, false, nil)
		return
	}
	// Update backoff and clear pending state
	grm.operations.updateOperation(key, false, true, err)

	// Log error
	klog.Errorf("%v", existingOp.expBackoff.
		GenerateNoRetriesPermittedMsg(fmt.Sprintf("%+v", key)))
}

func (grm *nestedPendingOperations) Wait() {
	grm.lock.Lock()
	defer grm.lock.Unlock()

	for grm.operations.len() > 0 {
		grm.cond.Wait()
	}
}

func (o *operationsCache) addPending(opKey operationKey) {
	// Add pending record
	operName := opKey.operationName
	if len(operName) == 0 {
		operName = EmptyOperationName
	}
	if _, ok := o.pendingOps[opKey.volumeName]; !ok {
		o.pendingOps[opKey.volumeName] = &volumeRelatedOperation{
			operationName: "",
			nodeRelated:   make(map[types.NodeName]*nodeRelatedOperation),
		}
	}
	vro := o.pendingOps[opKey.volumeName]
	if opKey.podName != EmptyUniquePodName || opKey.nodeName != EmptyNodeName {
		if _, ok := vro.nodeRelated[opKey.nodeName]; !ok {
			vro.nodeRelated[opKey.nodeName] = &nodeRelatedOperation{
				operationName: "",
				podRelated:    make(map[volumetypes.UniquePodName]*podRelatedOperation),
			}
		}
		nro := vro.nodeRelated[opKey.nodeName]
		if opKey.podName != EmptyUniquePodName {
			pro := &podRelatedOperation{
				operationName: operName,
			}
			nro.podRelated[opKey.podName] = pro
			return
		}
		nro.operationName = operName
		return
	}
	// podName and nodeName both empty
	vro.operationName = operName
}

func (o *operationsCache) addOperation(op *operation) {
	// Assumes lock has been acquired by caller.
	opKey := op.key
	// Add operation
	o.allOps[opKey] = op
	// Add pending recored
	o.addPending(opKey)
}

func (o *operationsCache) getOperation(opKey operationKey) (*operation, error) {
	// Assumes lock has been acquired by caller.
	if op, ok := o.allOps[opKey]; ok {
		return op, nil
	}
	return nil, fmt.Errorf("operation %+v not found", opKey)
}

func (o *operationsCache) deletePending(opKey operationKey) {
	// Remove pending records
	if volumePendingOp, ok := o.pendingOps[opKey.volumeName]; ok {
		if opKey.podName != EmptyUniquePodName {
			if nodePendingOp, ok := volumePendingOp.nodeRelated[opKey.nodeName]; ok {
				delete(nodePendingOp.podRelated, opKey.podName)
				if len(nodePendingOp.podRelated) == 0 {
					delete(volumePendingOp.nodeRelated, opKey.nodeName)
				}
			}
		} else if opKey.nodeName != EmptyNodeName {
			delete(volumePendingOp.nodeRelated, opKey.nodeName)
		} else {
			volumePendingOp.operationName = ""
		}
		// Cleanup pendings
		if len(volumePendingOp.operationName) == 0 && len(volumePendingOp.nodeRelated) == 0 {
			delete(o.pendingOps, opKey.volumeName)
		}
	}
}

func (o *operationsCache) deleteOperation(opKey operationKey) {
	// Assumes lock has been acquired by caller.

	// Remove pending records
	o.deletePending(opKey)
	// Remove operation
	delete(o.allOps, opKey)
}

func (o *operationsCache) updateOperation(opKey operationKey, isPending, isUpdateBackoff bool, backOffErr *error) {
	// Assumes lock has been acquired by caller.
	if op, ok := o.allOps[opKey]; ok {
		if isUpdateBackoff {
			op.expBackoff.Update(backOffErr)
		}
		op.operationPending = isPending
	}
	if isPending {
		o.addPending(opKey)
	} else {
		o.deletePending(opKey)
	}
}

func (o *operationsCache) isOperationExists(opKey operationKey) (bool, *operation) {
	// Assumes lock has been acquired by caller.
	if op, ok := o.allOps[opKey]; ok {
		return ok, op
	}
	return false, nil
}

func (o *operationsCache) hasOperationPending(volumeName v1.UniqueVolumeName,
	podName volumetypes.UniquePodName,
	nodeName types.NodeName) bool {
	// Assumes lock has been acquired by caller.
	if volumeName == EmptyUniqueVolumeName {
		// volumeName empty for verifyVolumesAreAttachedPerNode
		// which can running concurrently
		return false
	}
	vro, exists := o.pendingOps[volumeName]
	if !exists {
		return false
	}

	if len(vro.operationName) > 0 {
		// previous operation with node and pod both empty
		return true
	}
	if podName != EmptyUniquePodName || nodeName != EmptyNodeName {
		if nro, ok := vro.nodeRelated[nodeName]; ok {
			if len(nro.operationName) > 0 {
				return true
			}
			if podName != EmptyUniquePodName {
				if _, ok := nro.podRelated[podName]; ok {
					return true
				}
			} else {
				if len(nro.podRelated) > 0 {
					return true
				}
			}
		}
	} else {
		// pod empty and node empty
		if len(vro.nodeRelated) > 0 {
			return true
		}
	}

	return false
}

func (o *operationsCache) len() int {
	return len(o.allOps)
}

// NewAlreadyExistsError returns a new instance of AlreadyExists error.
func NewAlreadyExistsError(key operationKey) error {
	return alreadyExistsError{key}
}

// IsAlreadyExists returns true if an error returned from
// NestedPendingOperations indicates a new operation can not be started because
// an operation with the same operation name is already executing.
func IsAlreadyExists(err error) bool {
	switch err.(type) {
	case alreadyExistsError:
		return true
	default:
		return false
	}
}

// alreadyExistsError is the error returned by NestedPendingOperations when a
// new operation can not be started because an operation with the same operation
// name is already executing.
type alreadyExistsError struct {
	operationKey operationKey
}

var _ error = alreadyExistsError{}

func (err alreadyExistsError) Error() string {
	return fmt.Sprintf(
		"Failed to create operation with name %+v. An operation with that name is already executing.",
		err.operationKey)
}
