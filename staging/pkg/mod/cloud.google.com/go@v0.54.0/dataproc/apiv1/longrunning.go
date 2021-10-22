// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package dataproc

import (
	"context"

	gax "github.com/googleapis/gax-go/v2"
)

// Cancel starts asynchronous cancellation on a long-running operation.
// The server makes a best effort to cancel the operation, but success is not guaranteed.
// Clients can use Poll or other methods to check whether the cancellation succeeded or whether the operation
// completed despite cancellation. On successful cancellation, the operation is not deleted;
// instead, op.Poll returns an error with code Canceled.
func (op *InstantiateInlineWorkflowTemplateOperation) Cancel(ctx context.Context, opts ...gax.CallOption) error {
	return op.lro.Cancel(ctx, opts...)
}

// Delete deletes a long-running operation.
// This method indicates that the client is no longer interested in the operation result.
// It does not cancel the operation.
func (op *InstantiateInlineWorkflowTemplateOperation) Delete(ctx context.Context, opts ...gax.CallOption) error {
	return op.lro.Delete(ctx, opts...)
}

// Cancel starts asynchronous cancellation on a long-running operation.
// The server makes a best effort to cancel the operation, but success is not guaranteed.
// Clients can use Poll or other methods to check whether the cancellation succeeded or whether the operation
// completed despite cancellation. On successful cancellation, the operation is not deleted;
// instead, op.Poll returns an error with code Canceled.
func (op *InstantiateWorkflowTemplateOperation) Cancel(ctx context.Context, opts ...gax.CallOption) error {
	return op.lro.Cancel(ctx, opts...)
}

// Delete deletes a long-running operation.
// This method indicates that the client is no longer interested in the operation result.
// It does not cancel the operation.
func (op *InstantiateWorkflowTemplateOperation) Delete(ctx context.Context, opts ...gax.CallOption) error {
	return op.lro.Delete(ctx, opts...)
}

// Delete deletes a long-running operation.
// This method indicates that the client is no longer interested in the operation result.
// It does not cancel the operation.
func (op *CreateClusterOperation) Delete(ctx context.Context, opts ...gax.CallOption) error {
	return op.lro.Delete(ctx, opts...)
}

// Delete deletes a long-running operation.
// This method indicates that the client is no longer interested in the operation result.
// It does not cancel the operation.
func (op *DeleteClusterOperation) Delete(ctx context.Context, opts ...gax.CallOption) error {
	return op.lro.Delete(ctx, opts...)
}

// Delete deletes a long-running operation.
// This method indicates that the client is no longer interested in the operation result.
// It does not cancel the operation.
func (op *DiagnoseClusterOperation) Delete(ctx context.Context, opts ...gax.CallOption) error {
	return op.lro.Delete(ctx, opts...)
}

// Delete deletes a long-running operation.
// This method indicates that the client is no longer interested in the operation result.
// It does not cancel the operation.
func (op *UpdateClusterOperation) Delete(ctx context.Context, opts ...gax.CallOption) error {
	return op.lro.Delete(ctx, opts...)
}
