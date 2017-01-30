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

package util

import (
	"fmt"
	"time"

	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

// Type of the operation that can be executed in Federated.
type FederatedOperationType string

const (
	OperationTypeAdd    = "add"
	OperationTypeUpdate = "update"
	OperationTypeDelete = "delete"
)

// FederatedOperation definition contains type (add/update/delete) and the object itself.
type FederatedOperation struct {
	Type        FederatedOperationType
	ClusterName string
	Obj         pkgruntime.Object
}

// A helper that executes the given set of updates on federation, in parallel.
type FederatedUpdater interface {
	// Executes the given set of operations within the specified timeout.
	// Timeout is best-effort. There is no guarantee that the underlying operations are
	// stopped when it is reached. However the function will return after the timeout
	// with a non-nil error.
	Update([]FederatedOperation, time.Duration) error

	UpdateWithOnError([]FederatedOperation, time.Duration, func(FederatedOperation, error)) error
}

// A function that executes some operation using the passed client and object.
type FederatedOperationHandler func(kubeclientset.Interface, pkgruntime.Object) error

type federatedUpdaterImpl struct {
	federation FederationView

	addFunction    FederatedOperationHandler
	updateFunction FederatedOperationHandler
	deleteFunction FederatedOperationHandler
}

func NewFederatedUpdater(federation FederationView, add, update, del FederatedOperationHandler) FederatedUpdater {
	return &federatedUpdaterImpl{
		federation:     federation,
		addFunction:    add,
		updateFunction: update,
		deleteFunction: del,
	}
}

func (fu *federatedUpdaterImpl) Update(ops []FederatedOperation, timeout time.Duration) error {
	return fu.UpdateWithOnError(ops, timeout, nil)
}

func (fu *federatedUpdaterImpl) UpdateWithOnError(ops []FederatedOperation, timeout time.Duration, onError func(FederatedOperation, error)) error {
	done := make(chan error, len(ops))
	for _, op := range ops {
		go func(op FederatedOperation) {
			clusterName := op.ClusterName

			// TODO: Ensure that the clientset has reasonable timeout.
			clientset, err := fu.federation.GetClientsetForCluster(clusterName)
			if err != nil {
				done <- err
				return
			}

			switch op.Type {
			case OperationTypeAdd:
				err = fu.addFunction(clientset, op.Obj)
			case OperationTypeUpdate:
				err = fu.updateFunction(clientset, op.Obj)
			case OperationTypeDelete:
				err = fu.deleteFunction(clientset, op.Obj)
			}
			if err != nil && onError != nil {
				onError(op, err)
			}
			done <- err
		}(op)
	}
	start := time.Now()
	for i := 0; i < len(ops); i++ {
		now := time.Now()
		if !now.Before(start.Add(timeout)) {
			return fmt.Errorf("failed to finish all operations in %v", timeout)
		}
		select {
		case err := <-done:
			if err != nil {
				return err
			}
		case <-time.After(start.Add(timeout).Sub(now)):
			return fmt.Errorf("failed to finish all operations in %v", timeout)
		}
	}
	// All operations finished in time.
	return nil
}
