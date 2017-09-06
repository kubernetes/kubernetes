/*
Copyright 2017 The Kubernetes Authors.

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

package gce

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
)

const (
	// Key used to persist UIDs to configmaps.
	UIDConfigMapName = "ingress-uid"
	// Namespace which contains the above config map
	UIDNamespace = metav1.NamespaceSystem
	// Data keys for the specific ids
	UIDCluster     = "uid"
	UIDProvider    = "provider-uid"
	UIDLengthBytes = 8
	// Frequency of the updateFunc event handler being called
	// This does not actually query the apiserver for current state - the local cache value is used.
	updateFuncFrequency = 10 * time.Minute
)

type ClusterID struct {
	idLock     sync.RWMutex
	client     clientset.Interface
	cfgMapKey  string
	store      cache.Store
	providerID *string
	clusterID  *string
}

// Continually watches for changes to the cluster id config map
func (gce *GCECloud) watchClusterID() {
	gce.ClusterID = ClusterID{
		cfgMapKey: fmt.Sprintf("%v/%v", UIDNamespace, UIDConfigMapName),
		client:    gce.client,
	}

	mapEventHandler := cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			m, ok := obj.(*v1.ConfigMap)
			if !ok || m == nil {
				glog.Errorf("Expected v1.ConfigMap, item=%+v, typeIsOk=%v", obj, ok)
				return
			}
			if m.Namespace != UIDNamespace ||
				m.Name != UIDConfigMapName {
				return
			}

			glog.V(4).Infof("Observed new configmap for clusteriD: %v, %v; setting local values", m.Name, m.Data)
			gce.ClusterID.update(m)
		},
		UpdateFunc: func(old, cur interface{}) {
			m, ok := cur.(*v1.ConfigMap)
			if !ok || m == nil {
				glog.Errorf("Expected v1.ConfigMap, item=%+v, typeIsOk=%v", cur, ok)
				return
			}

			if m.Namespace != UIDNamespace ||
				m.Name != UIDConfigMapName {
				return
			}

			if reflect.DeepEqual(old, cur) {
				return
			}

			glog.V(4).Infof("Observed updated configmap for clusteriD %v, %v; setting local values", m.Name, m.Data)
			gce.ClusterID.update(m)
		},
	}

	listerWatcher := cache.NewListWatchFromClient(gce.ClusterID.client.Core().RESTClient(), "configmaps", UIDNamespace, fields.Everything())
	var controller cache.Controller
	gce.ClusterID.store, controller = cache.NewInformer(newSingleObjectListerWatcher(listerWatcher, UIDConfigMapName), &v1.ConfigMap{}, updateFuncFrequency, mapEventHandler)

	controller.Run(nil)
}

// GetID returns the id which is unique to this cluster
// if federated, return the provider id (unique to the cluster)
// if not federated, return the cluster id
func (ci *ClusterID) GetID() (string, error) {
	if err := ci.getOrInitialize(); err != nil {
		return "", err
	}

	ci.idLock.RLock()
	defer ci.idLock.RUnlock()
	if ci.clusterID == nil {
		return "", errors.New("Could not retrieve cluster id")
	}

	// If provider ID is set, (Federation is enabled) use this field
	if ci.providerID != nil {
		return *ci.providerID, nil
	}

	// providerID is not set, use the cluster id
	return *ci.clusterID, nil
}

// GetFederationId returns the id which could represent the entire Federation
// or just the cluster if not federated.
func (ci *ClusterID) GetFederationId() (string, bool, error) {
	if err := ci.getOrInitialize(); err != nil {
		return "", false, err
	}

	ci.idLock.RLock()
	defer ci.idLock.RUnlock()
	if ci.clusterID == nil {
		return "", false, errors.New("Could not retrieve cluster id")
	}

	// If provider ID is not set, return false
	if ci.providerID == nil || *ci.clusterID == *ci.providerID {
		return "", false, nil
	}

	return *ci.clusterID, true, nil
}

// getOrInitialize either grabs the configmaps current value or defines the value
// and sets the configmap. This is for the case of the user calling GetClusterID()
// before the watch has begun.
func (ci *ClusterID) getOrInitialize() error {
	if ci.store == nil {
		return errors.New("GCECloud.ClusterID is not ready. Call Initialize() before using.")
	}

	if ci.clusterID != nil {
		return nil
	}

	exists, err := ci.getConfigMap()
	if err != nil {
		return err
	} else if exists {
		return nil
	}

	// The configmap does not exist - let's try creating one.
	newId, err := makeUID()
	if err != nil {
		return err
	}

	glog.V(4).Infof("Creating clusteriD: %v", newId)
	cfg := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      UIDConfigMapName,
			Namespace: UIDNamespace,
		},
	}
	cfg.Data = map[string]string{
		UIDCluster:  newId,
		UIDProvider: newId,
	}

	if _, err := ci.client.Core().ConfigMaps(UIDNamespace).Create(cfg); err != nil {
		glog.Errorf("GCE cloud provider failed to create %v config map to store cluster id: %v", ci.cfgMapKey, err)
		return err
	}

	glog.V(2).Infof("Created a config map containing clusteriD: %v", newId)
	ci.update(cfg)
	return nil
}

func (ci *ClusterID) getConfigMap() (bool, error) {
	item, exists, err := ci.store.GetByKey(ci.cfgMapKey)
	if err != nil {
		return false, err
	}
	if !exists {
		return false, nil
	}

	m, ok := item.(*v1.ConfigMap)
	if !ok || m == nil {
		err = fmt.Errorf("Expected v1.ConfigMap, item=%+v, typeIsOk=%v", item, ok)
		glog.Error(err)
		return false, err
	}
	ci.update(m)
	return true, nil
}

func (ci *ClusterID) update(m *v1.ConfigMap) {
	ci.idLock.Lock()
	defer ci.idLock.Unlock()
	if clusterID, exists := m.Data[UIDCluster]; exists {
		ci.clusterID = &clusterID
	}
	if provId, exists := m.Data[UIDProvider]; exists {
		ci.providerID = &provId
	}
}

func makeUID() (string, error) {
	b := make([]byte, UIDLengthBytes)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func newSingleObjectListerWatcher(lw cache.ListerWatcher, objectName string) *singleObjListerWatcher {
	return &singleObjListerWatcher{lw: lw, objectName: objectName}
}

type singleObjListerWatcher struct {
	lw         cache.ListerWatcher
	objectName string
}

func (sow *singleObjListerWatcher) List(options metav1.ListOptions) (runtime.Object, error) {
	options.FieldSelector = "metadata.name=" + sow.objectName
	return sow.lw.List(options)
}

func (sow *singleObjListerWatcher) Watch(options metav1.ListOptions) (watch.Interface, error) {
	options.FieldSelector = "metadata.name=" + sow.objectName
	return sow.lw.Watch(options)
}
