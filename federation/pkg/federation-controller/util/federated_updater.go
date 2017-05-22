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
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
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
	Key         string
}

// A helper that executes the given set of updates on federation, in parallel.
type FederatedUpdater interface {
	// Executes the given set of operations.
	Update([]FederatedOperation) error
}

// A function that executes some operation using the passed client and object.
type FederatedOperationHandler func(kubeclientset.Interface, pkgruntime.Object) error

type federatedUpdaterImpl struct {
	federation FederationView

	kind string

	timeout time.Duration

	eventRecorder record.EventRecorder

	addFunction    FederatedOperationHandler
	updateFunction FederatedOperationHandler
	deleteFunction FederatedOperationHandler
}

func NewFederatedUpdater(federation FederationView, kind string, timeout time.Duration, recorder record.EventRecorder, add, update, del FederatedOperationHandler) FederatedUpdater {
	return &federatedUpdaterImpl{
		federation:     federation,
		kind:           kind,
		timeout:        timeout,
		eventRecorder:  recorder,
		addFunction:    add,
		updateFunction: update,
		deleteFunction: del,
	}
}

func (fu *federatedUpdaterImpl) recordEvent(obj runtime.Object, eventType, eventVerb string, args ...interface{}) {
	messageFmt := eventVerb + " %s %q in cluster %s"
	fu.eventRecorder.Eventf(obj, api.EventTypeNormal, eventType, messageFmt, args...)
}

// Update executes the given set of operations within the timeout specified for
// the instance. Timeout is best-effort. There is no guarantee that the
// underlying operations are stopped when it is reached. However the function
// will return after the timeout with a non-nil error.
func (fu *federatedUpdaterImpl) Update(ops []FederatedOperation) error {
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

			eventArgs := []interface{}{fu.kind, op.Key, clusterName}
			baseEventType := fmt.Sprintf("%s", op.Type)
			eventType := fmt.Sprintf("%sInCluster", strings.Title(baseEventType))

			switch op.Type {
			case OperationTypeAdd:
				// TODO s+OperationTypeAdd+OperationTypeCreate+
				baseEventType = "create"
				eventType := "CreateInCluster"

				fu.recordEvent(op.Obj, eventType, "Creating", eventArgs...)
				err = fu.addFunction(clientset, op.Obj)
			case OperationTypeUpdate:
				fu.recordEvent(op.Obj, eventType, "Updating", eventArgs...)
				err = fu.updateFunction(clientset, op.Obj)
			case OperationTypeDelete:
				fu.recordEvent(op.Obj, eventType, "Deleting", eventArgs...)
				err = fu.deleteFunction(clientset, op.Obj)
				// IsNotFound error is fine since that means the object is deleted already.
				if errors.IsNotFound(err) {
					err = nil
				}
			}

			if err != nil {
				eventType := eventType + "Failed"
				messageFmt := "Failed to " + baseEventType + " %s %q in cluster %s: %v"
				eventArgs = append(eventArgs, err)
				fu.eventRecorder.Eventf(op.Obj, api.EventTypeWarning, eventType, messageFmt, eventArgs...)
			}

			done <- err
		}(op)
	}
	start := time.Now()
	for i := 0; i < len(ops); i++ {
		now := time.Now()
		if !now.Before(start.Add(fu.timeout)) {
			return fmt.Errorf("failed to finish all operations in %v", fu.timeout)
		}
		select {
		case err := <-done:
			if err != nil {
				return err
			}
		case <-time.After(start.Add(fu.timeout).Sub(now)):
			return fmt.Errorf("failed to finish all operations in %v", fu.timeout)
		}
	}
	// All operations finished in time.
	return nil
}
