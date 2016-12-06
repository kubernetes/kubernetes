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

package registry

import (
	"reflect"

	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/watch"

	//kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	etcd "github.com/coreos/etcd/client"
	"github.com/golang/glog"
	api_v1 "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	etcdutil "k8s.io/kubernetes/pkg/storage/etcd/util"
	"strings"
)

const (
	userAgentName = "federation-controller"
)

type podStore struct {
	store       storage.Interface
	etcdKeysAPI etcd.KeysAPI
}

func NewPodStore(client etcd.Client, c storage.Interface) *podStore {
	return &podStore{
		store:       c,
		etcdKeysAPI: etcd.NewKeysAPI(client),
	}
}

// Versioner implements storage.Interface.Versioner.
func (s *podStore) Versioner() storage.Versioner {
	return s.store.Versioner()
}

// Get implements storage.Interface.Get.
func (s *podStore) Get(ctx context.Context, key string, resourceVersion string, out runtime.Object, ignoreNotFound bool) error {
	return s.store.Get(ctx, key, resourceVersion, out, ignoreNotFound)
}

// Create implements storage.Interface.Create.
func (s *podStore) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	return s.store.Create(ctx, key, obj, out, ttl)
}

// Delete implements storage.Interface.Delete.
func (s *podStore) Delete(ctx context.Context, key string, out runtime.Object, precondtions *storage.Preconditions) error {
	return s.store.Delete(ctx, key, out, precondtions)
}

// GuaranteedUpdate implements storage.Interface.GuaranteedUpdate.
func (s *podStore) GuaranteedUpdate(ctx context.Context, key string, out runtime.Object, ignoreNotFound bool, precondtions *storage.Preconditions, tryUpdate storage.UpdateFunc, suggestion ...runtime.Object) error {
	return s.store.GuaranteedUpdate(ctx, key, out, ignoreNotFound, precondtions, tryUpdate, suggestion...)
}

// GetToList implements storage.Interface.GetToList.
func (s *podStore) GetToList(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	return s.store.GetToList(ctx, key, resourceVersion, pred, listObj)
}

// List implements storage.Interface.List.
func (s *podStore) List(ctx context.Context, key, resourceVersion string, pred storage.SelectionPredicate, listObj runtime.Object) error {
	listPtr, err := meta.GetItemsPtr(listObj)
	if err != nil {
		return err
	}
	vPtr, err := conversion.EnforcePtr(listPtr)
	if err != nil || vPtr.Kind() != reflect.Slice {
		// This should not happen at runtime.
		panic("need ptr to slice")
	}
	// get clusters from etcd
	clusters, _, _ := s.listEtcdNode(ctx, "/registry/clusters")

	for _, node := range clusters {
		if node.Value != "" {
			v := node.Value
			glog.Infof("value %v", v)
			ip := strings.Split(strings.Split(v, "\"serverAddress\":\"")[1], "\"}]}")[0]
			// get the list of pods from the clusters
			clusterConfig, err := clientcmd.BuildConfigFromFlags(ip, "")
			if err == nil && clusterConfig != nil {
				clientset := kubeclientset.NewForConfigOrDie(restclient.AddUserAgent(clusterConfig, userAgentName))
				tmpPodList, _ := clientset.Core().Pods(api_v1.NamespaceAll).List(api_v1.ListOptions{})
				for i := range tmpPodList.Items {
					p := tmpPodList.Items[i]
					vPtr.Set(reflect.Append(vPtr, reflect.ValueOf(p)))
				}
			}
		}
	}
	return nil
}

func (s *podStore) listEtcdNode(ctx context.Context, key string) ([]*etcd.Node, uint64, error) {
	if ctx == nil {
		glog.Errorf("Context is nil")
	}
	opts := etcd.GetOptions{
		Recursive: true,
		Sort:      true,
	}
	result, err := s.etcdKeysAPI.Get(ctx, key, &opts)
	if err != nil {
		var index uint64
		if etcdError, ok := err.(etcd.Error); ok {
			index = etcdError.Index
		}
		nodes := make([]*etcd.Node, 0)
		if etcdutil.IsEtcdNotFound(err) {
			return nodes, index, nil
		} else {
			return nodes, index, toStorageErr(err, key, 0)
		}
	}
	return result.Node.Nodes, result.Index, nil
}

func toStorageErr(err error, key string, rv int64) error {
	if err == nil {
		return nil
	}
	switch {
	case etcdutil.IsEtcdNotFound(err):
		return storage.NewKeyNotFoundError(key, rv)
	case etcdutil.IsEtcdNodeExist(err):
		return storage.NewKeyExistsError(key, rv)
	case etcdutil.IsEtcdTestFailed(err):
		return storage.NewResourceVersionConflictsError(key, rv)
	case etcdutil.IsEtcdUnreachable(err):
		return storage.NewUnreachableError(key, rv)
	default:
		return err
	}
}

// Watch implements storage.Interface.Watch.
func (s *podStore) Watch(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate) (watch.Interface, error) {
	watchInterface, err := s.store.Watch(ctx, key, resourceVersion, pred)
	return watchInterface, err
}

// WatchList implements storage.Interface.WatchList.
func (s *podStore) WatchList(ctx context.Context, key string, resourceVersion string, pred storage.SelectionPredicate) (watch.Interface, error) {
	return s.store.WatchList(ctx, key, resourceVersion, pred)
}
