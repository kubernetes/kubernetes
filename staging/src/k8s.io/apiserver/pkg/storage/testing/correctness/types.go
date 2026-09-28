/*
Copyright The Kubernetes Authors.

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

package correctness

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

// Operation represents a single operation recorded in a concurrent execution history.
type Operation struct {
	ClientID int
	Request  Request
	Response Response
	Start    time.Time
	End      time.Time
}

// Request represents an input invocation to the storage interface.
type Request struct {
	Op     OpType
	Key    string
	Create CreateRequest
	Get    GetRequest
	Delete DeleteRequest
	Update UpdateRequest
}

// CreateRequest contains parameters specific to Create operations.
type CreateRequest struct {
	Object runtime.Object
}

// GetRequest contains parameters specific to Get operations.
type GetRequest struct {
	Options storage.GetOptions
}

// DeleteRequest contains parameters specific to Delete operations.
type DeleteRequest struct {
	Preconditions *storage.Preconditions
}

// UpdateRequest contains parameters specific to Update / GuaranteedUpdate operations.
type UpdateRequest struct {
	UpdateFunc           storage.UpdateFunc
	IgnoreNotFound       bool
	Preconditions        *storage.Preconditions
	CachedExistingObject runtime.Object
}

// Describe formats the operation for debugging and visualization.
func (r Request) Describe(output Response) string {
	if output.Err != nil {
		switch {
		case storage.IsNotFound(output.Err):
			return fmt.Sprintf("%s(%s) -> Not Found", r.Op, r.Key)
		case storage.IsExist(output.Err):
			return fmt.Sprintf("%s(%s) -> Already Exists", r.Op, r.Key)
		case storage.IsConflict(output.Err):
			return fmt.Sprintf("%s(%s) -> Conflict", r.Op, r.Key)
		case storage.IsUnreachable(output.Err):
			return fmt.Sprintf("%s(%s) -> Unreachable", r.Op, r.Key)
		case storage.IsRequestTimeout(output.Err):
			return fmt.Sprintf("%s(%s) -> Timeout", r.Op, r.Key)
		case storage.IsInvalidObj(output.Err):
			errStr := output.Err.Error()
			errParts := strings.Split(errStr, "Precondition failed:")
			if len(errParts) > 1 {
				errStr = errParts[1]
			}
			return fmt.Sprintf("%s(%s) -> Invalid %s", r.Op, r.Key, errStr)
		case storage.IsCorruptObject(output.Err):
			return fmt.Sprintf("%s(%s) -> Corrupt", r.Op, r.Key)
		default:
			return fmt.Sprintf("%s(%s) -> %v", r.Op, r.Key, output.Err)
		}
	}
	accessor, err := meta.Accessor(output.Object)
	if err != nil {
		panic(err)
	}
	switch r.Op {
	case OpCreate:
		return fmt.Sprintf("%s(%s) -> RV: %s, UID: %s", r.Op, r.Key, accessor.GetResourceVersion(), accessor.GetUID())
	case OpDelete:
		if r.Delete.Preconditions != nil {
			if r.Delete.Preconditions.ResourceVersion != nil && *r.Delete.Preconditions.ResourceVersion != "" {
				return fmt.Sprintf("%s(if RV(%s) ==%s) -> Deleted", r.Op, r.Key, *r.Delete.Preconditions.ResourceVersion)
			}
			if r.Delete.Preconditions.UID != nil && *r.Delete.Preconditions.UID != "" {
				return fmt.Sprintf("%s(if UID(%s) == %s) -> Deleted", r.Op, r.Key, *r.Delete.Preconditions.UID)
			}
		}
		return fmt.Sprintf("%s(%s) -> Deleted", r.Op, r.Key)
	case OpGet:
		return fmt.Sprintf("%s(%s) -> RV: %s, UID: %s", r.Op, r.Key, accessor.GetResourceVersion(), accessor.GetUID())
	case OpUpdate:
		return fmt.Sprintf("%s(%s) -> RV: %s, UID: %s", r.Op, r.Key, accessor.GetResourceVersion(), accessor.GetUID())
	default:
		return fmt.Sprintf("%s(%s) -> RV: %s", r.Op, r.Key, accessor.GetResourceVersion())
	}
}

// OpType identifies the storage interface operation.
type OpType string

const (
	OpCreate OpType = "Create"
	OpDelete OpType = "Delete"
	OpGet    OpType = "Get"
	OpUpdate OpType = "Update"
)

// Response represents the output/result from the storage interface invocation.
type Response struct {
	Object runtime.Object
	Err    error
}

// WatchRequest contains parameters for a watch stream.
type WatchRequest struct {
	ResourceVersion string
}

// WatchResponse contains the events and any terminal error received from a watch stream.
type WatchResponse struct {
	Events []watch.Event
	Err    error
}

// WatchOperation captures a recorded watch operation with its request and response.
type WatchOperation struct {
	Request  WatchRequest
	Response WatchResponse
}
