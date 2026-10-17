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

package direct

import (
	"context"
	"fmt"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

// Storage is the subset of a resource's REST storage that direct listers read from.
// It is implemented by *genericregistry.Store, which serves reads from the watch cache
// when the request's resourceVersion is "0".
type Storage interface {
	Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error)
	List(ctx context.Context, options *metainternalversion.ListOptions) (runtime.Object, error)
}

// Client reads objects from a resource's watch cache, converting the internal objects
// returned by the storage layer into the external version listers hand to their consumers.
type Client interface {
	Get(ctx context.Context, namespace, name string, options metav1.GetOptions) (runtime.Object, error)
	List(ctx context.Context, namespace string, options metav1.ListOptions) (runtime.Object, error)
}

type storageClient struct {
	factory *directSharedInformerFactory
	gvr     schema.GroupVersionResource
}

var _ Client = &storageClient{}

func (c *storageClient) Get(ctx context.Context, namespace, name string, options metav1.GetOptions) (runtime.Object, error) {
	storage, err := c.storage()
	if err != nil {
		return nil, err
	}
	obj, err := storage.Get(c.newContext(ctx, namespace, "get"), name, &options)
	if err != nil {
		return nil, err
	}
	return c.toExternal(obj)
}

func (c *storageClient) List(ctx context.Context, namespace string, options metav1.ListOptions) (runtime.Object, error) {
	storage, err := c.storage()
	if err != nil {
		return nil, err
	}
	internalOptions := &metainternalversion.ListOptions{}
	if err := metainternalversion.Convert_v1_ListOptions_To_internalversion_ListOptions(&options, internalOptions, nil); err != nil {
		return nil, err
	}
	obj, err := storage.List(c.newContext(ctx, namespace, "list"), internalOptions)
	if err != nil {
		return nil, err
	}
	return c.toExternal(obj)
}

func (c *storageClient) storage() (Storage, error) {
	storage := c.factory.storageFor(c.gvr.GroupResource())
	if storage == nil {
		return nil, fmt.Errorf("watch cache storage for %s is not yet initialized", c.gvr.GroupResource())
	}
	return storage, nil
}

// newContext layers the namespace and request info the storage layer expects onto the
// caller's context, so cancellation, deadlines and tracing spans of the request that
// triggered the read still apply to it.
func (c *storageClient) newContext(ctx context.Context, namespace, verb string) context.Context {
	ctx = genericapirequest.WithNamespace(ctx, namespace)
	return genericapirequest.WithRequestInfo(ctx, &genericapirequest.RequestInfo{
		IsResourceRequest: true,
		Verb:              verb,
		APIGroup:          c.gvr.Group,
		APIVersion:        c.gvr.Version,
		Resource:          c.gvr.Resource,
		Namespace:         namespace,
	})
}

// toExternal converts the internal object returned by the storage layer to the external
// version, and clears the kind so objects match what a client-go lister returns.
func (c *storageClient) toExternal(obj runtime.Object) (runtime.Object, error) {
	out, err := legacyscheme.Scheme.ConvertToVersion(obj, c.gvr.GroupVersion())
	if err != nil {
		return nil, err
	}
	out.GetObjectKind().SetGroupVersionKind(schema.GroupVersionKind{})
	return out, nil
}
