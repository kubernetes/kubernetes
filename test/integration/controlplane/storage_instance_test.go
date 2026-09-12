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

package controlplane

import (
	"maps"
	"slices"
	"strings"
	"sync"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	"k8s.io/client-go/tools/cache"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

type countingRESTOptionsGetter struct {
	delegate generic.RESTOptionsGetter

	lock   sync.Mutex
	counts map[schema.GroupResource]int
}

func (c *countingRESTOptionsGetter) GetRESTOptions(resource schema.GroupResource, example runtime.Object) (generic.RESTOptions, error) {
	opts, err := c.delegate.GetRESTOptions(resource, example)
	if err != nil {
		return opts, err
	}

	decorator := opts.Decorator
	opts.Decorator = func(
		config *storagebackend.ConfigForResource,
		resourcePrefix string,
		cacheKeyFunc func(obj runtime.Object) (string, error),
		newFunc func() runtime.Object,
		newListFunc func() runtime.Object,
		getAttrsFunc storage.AttrFunc,
		trigger storage.IndexerFuncs,
		indexers *cache.Indexers) (storage.Interface, factory.DestroyFunc, error) {
		c.lock.Lock()
		c.counts[resource]++
		c.lock.Unlock()
		return decorator(config, resourcePrefix, cacheKeyFunc, newFunc, newListFunc, getAttrsFunc, trigger, indexers)
	}
	return opts, nil
}

func (c *countingRESTOptionsGetter) snapshot() map[schema.GroupResource]int {
	c.lock.Lock()
	defer c.lock.Unlock()

	return maps.Clone(c.counts)
}

// knownDuplicateStorageInstances allowlists resources that are known to build
// more than one storage instance today, so this test does not block unrelated
// changes. Entries should only ever be removed, never added.
// Kube-apiserver built-in resource only.
// See https://github.com/kubernetes/kubernetes/issues/133877.
var knownDuplicateStorageInstances = map[schema.GroupResource]int{
	{Group: "", Resource: "events"}:          2,
	{Group: "", Resource: "serviceaccounts"}: 2,
}

func TestSingleStorageInstancePerResource(t *testing.T) {
	counter := &countingRESTOptionsGetter{counts: map[schema.GroupResource]int{}}

	instanceOptions := kubeapiservertesting.NewDefaultTestServerOptions()
	instanceOptions.RESTOptionsGetterWrapFunc = func(delegate generic.RESTOptionsGetter) generic.RESTOptionsGetter {
		counter.delegate = delegate
		return counter
	}

	server := kubeapiservertesting.StartTestServerOrDie(t, instanceOptions, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	counts := counter.snapshot()
	if len(counts) == 0 {
		t.Fatalf("no store was observed; the counting RESTOptionsGetter is no longer wired into storage construction")
	}
	resources := slices.SortedFunc(maps.Keys(counts), func(a, b schema.GroupResource) int {
		return strings.Compare(a.String(), b.String())
	})

	for _, gr := range resources {
		want := 1
		known, isKnown := knownDuplicateStorageInstances[gr]
		if isKnown {
			want = known
		}
		got := counts[gr]
		switch {
		case got == want:
		case isKnown:
			t.Errorf("%s: built %d storage instances, expected the known-broken count of %d; update knownDuplicateStorageInstances", gr, got, want)
		default:
			t.Errorf("%s: built %d storage instances, expected exactly 1; a resource must not initialize its storage more than once (see https://github.com/kubernetes/kubernetes/issues/133877)", gr, got)
		}
	}

	allowlisted := slices.SortedFunc(maps.Keys(knownDuplicateStorageInstances), func(a, b schema.GroupResource) int {
		return strings.Compare(a.String(), b.String())
	})
	for _, gr := range allowlisted {
		if _, ok := counts[gr]; !ok {
			t.Errorf("%s: listed in knownDuplicateStorageInstances but no storage was built; remove the stale entry", gr)
		}
	}

}
