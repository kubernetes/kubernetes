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
	"bytes"
	"context"
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
	"k8s.io/apiserver/pkg/storage/value"
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

type prefixTransformer struct{}

func (prefixTransformer) TransformToStorage(_ context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	if dataCtx == nil {
		return nil, errors.New("missing value.Context in TransformToStorage")
	}
	prefix := append(append([]byte{}, dataCtx.AuthenticatedData()...), ':')
	return append(prefix, data...), nil
}

func (prefixTransformer) TransformFromStorage(_ context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	if dataCtx == nil {
		return nil, false, errors.New("missing value.Context in TransformFromStorage")
	}
	expectedPrefix := append(append([]byte{}, dataCtx.AuthenticatedData()...), ':')
	if !bytes.HasPrefix(data, expectedPrefix) {
		return nil, false, fmt.Errorf("authenticated data mismatch: expected prefix %q", string(expectedPrefix))
	}
	return bytes.TrimPrefix(data, expectedPrefix), false, nil
}

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

func setupStore(tCtx ktesting.TContext, transformer value.Transformer) (storage.Interface, string) {
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.WatchList, true)
	etcdURL, stop, err := framework.RunCustomEtcd(tCtx.Logger(), "storage_correctness_etcd", nil)
	if err != nil {
		tCtx.Fatalf("failed to start dedicated etcd: %v", err)
	}
	tCtx.Cleanup(stop)

	etcdConfig := storagebackend.NewDefaultConfig("registry", nil)
	etcdConfig.Transport.ServerList = []string{etcdURL}
	if transformer != nil {
		etcdConfig.Transformer = transformer
	}

	storageConfig := newEtcdStorageForResource(tCtx, etcdConfig, schema.GroupResource{Resource: "pods"})
	store, destroyFunc, err := generic.NewRawStorage(
		storageConfig,
		func() runtime.Object { return &api.Pod{} },
		func() runtime.Object { return &api.PodList{} },
		nil,
		"/pods",
	)
	if err != nil {
		tCtx.Fatalf("failed to create storage: %v", err)
	}
	tCtx.Cleanup(destroyFunc)

	return store, "/" + etcdConfig.Prefix
}

func setupCacher(tCtx ktesting.TContext, s storage.Interface) storage.Interface {
	storageConfig := newEtcdStorageForResource(tCtx, storagebackend.NewDefaultConfig("registry", nil), schema.GroupResource{Resource: "pods"})
	cacherConfig := cacher.Config{
		Storage:             s,
		Versioner:           storage.APIObjectVersioner{},
		GroupResource:       storageConfig.GroupResource,
		EventsHistoryWindow: cacher.DefaultEventFreshDuration,
		ResourcePrefix:      "/pods",
		KeyFunc:             cacheKeyFunc,
		NewFunc:             func() runtime.Object { return &api.Pod{} },
		NewListFunc:         func() runtime.Object { return &api.PodList{} },
		GetAttrsFunc:        registrypod.GetAttrs,
		IndexerFuncs:        map[string]storage.IndexerFunc{"spec.nodeName": registrypod.NodeNameTriggerFunc},
		Indexers:            registrypod.Indexers(),
		Codec:               storageConfig.Codec,
	}
	c, err := cacher.NewCacherFromConfig(cacherConfig)
	if err != nil {
		tCtx.Fatalf("failed to create cacher: %v", err)
	}
	delegator := cacher.NewCacheDelegator(c, s)
	tCtx.Cleanup(func() {
		delegator.Stop()
		c.Stop()
	})
	tCtx.Eventually(delegator.ReadinessCheck).WithPolling(10*time.Millisecond).WithTimeout(5*time.Second).Should(gomega.Succeed(), "storage ready")
	return delegator
}
