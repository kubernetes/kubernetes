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

package storage

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/server/options"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/kubeapiserver"
	registrypod "k8s.io/kubernetes/pkg/registry/core/pod"
	"k8s.io/kubernetes/test/integration/framework"
)

func cacheKeyFunc(obj runtime.Object) (string, error) {
	pod, ok := obj.(*api.Pod)
	if !ok {
		return "", fmt.Errorf("not a pod: %T", obj)
	}
	if len(pod.Namespace) == 0 {
		return "", errors.New("namespace cannot be empty for pods")
	}
	return "/pods/" + pod.Namespace + "/" + pod.Name, nil
}

func newEtcdStorageForResource(t *testing.T, etcdConfig *storagebackend.Config, resource schema.GroupResource) *storagebackend.ConfigForResource {
	t.Helper()

	completedConfig := kubeapiserver.NewStorageFactoryConfig().Complete(options.NewEtcdOptions(etcdConfig))
	completedConfig.APIResourceConfig = serverstorage.NewResourceConfig()
	factory, err := completedConfig.New()
	if err != nil {
		t.Fatalf("Error while making storage factory: %v", err)
	}
	resourceConfig, err := factory.NewConfig(resource, nil)
	if err != nil {
		t.Fatalf("Error while finding storage destination: %v", err)
	}
	return resourceConfig
}

func setupStore(t *testing.T, decorator generic.StorageDecorator) (storage.Interface, string) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchList, true)
	etcdURL, stop, err := framework.RunCustomEtcd(klog.FromContext(t.Context()), "storage_correctness_etcd", nil)
	if err != nil {
		t.Fatalf("failed to start dedicated etcd: %v", err)
	}
	t.Cleanup(stop)

	etcdConfig := storagebackend.NewDefaultConfig("registry", nil)
	etcdConfig.Transport.ServerList = []string{etcdURL}

	storageConfig := newEtcdStorageForResource(t, etcdConfig, schema.GroupResource{Resource: "pods"})
	storageConfig.EventsHistoryWindow = cacher.DefaultEventFreshDuration

	store, destroyFunc, err := decorator(
		storageConfig,
		"/pods",
		cacheKeyFunc,
		func() runtime.Object { return &api.Pod{} },
		func() runtime.Object { return &api.PodList{} },
		registrypod.GetAttrs,
		map[string]storage.IndexerFunc{"spec.nodeName": registrypod.NodeNameTriggerFunc},
		registrypod.Indexers(),
	)
	if err != nil {
		t.Fatalf("failed to create storage: %v", err)
	}
	t.Cleanup(destroyFunc)

	if rc, ok := store.(interface{ ReadinessCheck() error }); ok {
		err := wait.PollUntilContextTimeout(t.Context(), 10*time.Millisecond, 5*time.Second, true, func(ctx context.Context) (bool, error) {
			return rc.ReadinessCheck() == nil, nil
		})
		if err != nil {
			t.Fatalf("storage failed to become ready: %v", err)
		}
	}

	return store, "/" + etcdConfig.Prefix
}
