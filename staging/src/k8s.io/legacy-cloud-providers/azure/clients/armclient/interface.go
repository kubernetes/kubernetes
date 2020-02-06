// +build !providerless

/*
Copyright 2019 The Kubernetes Authors.

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

package armclient

import (
	"context"
	"net/http"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
	"k8s.io/legacy-cloud-providers/azure/retry"
)

// Interface is the client interface for ARM.
// Don't forget to run the following command to generate the mock client:
// mockgen -source=$GOPATH/src/k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers/azure/clients/armclient/interface.go -package=mockarmclient Interface > $GOPATH/src/k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers/azure/clients/armclient/mockarmclient/interface.go
type Interface interface {
	// Send sends a http request to ARM service with possible retry to regional ARM endpoint.
	Send(ctx context.Context, request *http.Request) (*http.Response, *retry.Error)

	// PreparePutRequest prepares put request
	PreparePutRequest(ctx context.Context, decorators ...autorest.PrepareDecorator) (*http.Request, error)

	// PreparePostRequest prepares post request
	PreparePostRequest(ctx context.Context, decorators ...autorest.PrepareDecorator) (*http.Request, error)

	// PrepareGetRequest prepares get request
	PrepareGetRequest(ctx context.Context, decorators ...autorest.PrepareDecorator) (*http.Request, error)

	// PrepareDeleteRequest preparse delete request
	PrepareDeleteRequest(ctx context.Context, decorators ...autorest.PrepareDecorator) (*http.Request, error)

	// PrepareHeadRequest prepares head request
	PrepareHeadRequest(ctx context.Context, decorators ...autorest.PrepareDecorator) (*http.Request, error)

	// WaitForAsyncOperationCompletion waits for an operation completion
	WaitForAsyncOperationCompletion(ctx context.Context, future *azure.Future, asyncOperationName string) error

	// WaitForAsyncOperationResult waits for an operation result.
	WaitForAsyncOperationResult(ctx context.Context, future *azure.Future, asyncOperationName string) (*http.Response, error)

	// SendAsync send a request and return a future object representing the async result as well as the origin http response
	SendAsync(ctx context.Context, request *http.Request) (*azure.Future, *http.Response, *retry.Error)

	// PutResource puts a resource by resource ID
	PutResource(ctx context.Context, resourceID string, parameters interface{}) (*http.Response, *retry.Error)

	// PutResourceWithDecorators puts a resource with decorators by resource ID
	PutResourceWithDecorators(ctx context.Context, resourceID string, parameters interface{}, decorators []autorest.PrepareDecorator) (*http.Response, *retry.Error)

	// PatchResource patches a resource by resource ID
	PatchResource(ctx context.Context, resourceID string, parameters interface{}) (*http.Response, *retry.Error)

	// PutResourceAsync puts a resource by resource ID in async mode
	PutResourceAsync(ctx context.Context, resourceID string, parameters interface{}) (*azure.Future, *retry.Error)

	// HeadResource heads a resource by resource ID
	HeadResource(ctx context.Context, resourceID string) (*http.Response, *retry.Error)

	// GetResource get a resource by resource ID
	GetResource(ctx context.Context, resourceID, expand string) (*http.Response, *retry.Error)

	//GetResourceWithDecorators get a resource with decorators by resource ID
	GetResourceWithDecorators(ctx context.Context, resourceID string, decorators []autorest.PrepareDecorator) (*http.Response, *retry.Error)

	// PostResource posts a resource by resource ID
	PostResource(ctx context.Context, resourceID, action string, parameters interface{}) (*http.Response, *retry.Error)

	// DeleteResource deletes a resource by resource ID
	DeleteResource(ctx context.Context, resourceID, ifMatch string) *retry.Error

	// DeleteResourceAsync delete a resource by resource ID and returns a future representing the async result
	DeleteResourceAsync(ctx context.Context, resourceID, ifMatch string) (*azure.Future, *retry.Error)

	// CloseResponse closes a response
	CloseResponse(ctx context.Context, response *http.Response)
}
