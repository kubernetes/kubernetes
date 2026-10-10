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
	"errors"
	"fmt"
	"time"

	"github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/server/options"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	api "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/kubeapiserver"
	registrypod "k8s.io/kubernetes/pkg/registry/core/pod"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"
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

func newEtcdStorageForResource(t testutils.TB, etcdConfig *storagebackend.Config, resource schema.GroupResource) *storagebackend.ConfigForResource {
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

func setupStore(tCtx ktesting.TContext, decorator generic.StorageDecorator) (storage.Interface, string) {
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.WatchList, true)
	etcdURL, stop, err := framework.RunCustomEtcd(tCtx.Logger(), "storage_correctness_etcd", nil)
	if err != nil {
		tCtx.Fatalf("failed to start dedicated etcd: %v", err)
	}
	tCtx.Cleanup(stop)

	etcdConfig := storagebackend.NewDefaultConfig("registry", nil)
	etcdConfig.Transport.ServerList = []string{etcdURL}

	storageConfig := newEtcdStorageForResource(tCtx, etcdConfig, schema.GroupResource{Resource: "pods"})
	storageConfig.EventsHistoryWindow = cacher.DefaultEventFreshDuration

	store, destroyFunc, err := decorator(
		storageConfig,
		"/pods",
		cacheKeyFunc,
		nil,
		func() runtime.Object { return &api.Pod{} },
		func() runtime.Object { return &api.PodList{} },
		registrypod.GetAttrs,
		map[string]storage.IndexerFunc{"spec.nodeName": registrypod.NodeNameTriggerFunc},
		registrypod.Indexers(),
	)
	if err != nil {
		tCtx.Fatalf("failed to create storage: %v", err)
	}
	tCtx.Cleanup(destroyFunc)

	if rc, ok := store.(interface{ ReadinessCheck() error }); ok {
		tCtx.Eventually(rc.ReadinessCheck).WithPolling(10*time.Millisecond).WithTimeout(5*time.Second).Should(gomega.Succeed(), "storage ready")
	}

	return store, "/" + etcdConfig.Prefix
}
