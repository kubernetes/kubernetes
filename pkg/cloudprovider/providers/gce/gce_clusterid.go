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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
)

const (
	// UIDConfigMapName is the Key used to persist UIDs to configmaps.
	UIDConfigMapName = "ingress-uid"

	// UIDNamespace is the namespace which contains the above config map
	UIDNamespace = metav1.NamespaceSystem

	// UIDCluster is the data keys for looking up the clusters UID
	UIDCluster = "uid"

	// UIDProvider is the data keys for looking up the providers UID
	UIDProvider = "provider-uid"

	// UIDLengthBytes is the length of a UID
	UIDLengthBytes = 8

	// Frequency of the updateFunc event handler being called
	// This does not actually query the apiserver for current state - the local cache value is used.
	updateFuncFrequency = 10 * time.Minute
)

// ClusterID is the struct for maintaining information about this cluster's ID
type ClusterID struct {
	idLock     sync.RWMutex
	client     clientset.Interface
	cfgMapKey  string
	store      cache.Store
	providerID *string
	clusterID  *string
}

// Continually watches for changes to the cluster id config map
func (g *Cloud) watchClusterID(stop <-chan struct{}) {
	g.ClusterID = ClusterID{
		cfgMapKey: fmt.Sprintf("%v/%v", UIDNamespace, UIDConfigMapName),
		client:    g.client,
	}

	mapEventHandler := cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			m, ok := obj.(*v1.ConfigMap)
			if !ok || m == nil {
				klog.Errorf("Expected v1.ConfigMap, item=%+v, typeIsOk=%v", obj, ok)
				return
			}
			if m.Namespace != UIDNamespace ||
				m.Name != UIDConfigMapName {
				return
			}

			klog.V(4).Infof("Observed new configmap for clusteriD: %v, %v; setting local values", m.Name, m.Data)
			g.ClusterID.update(m)
		},
		UpdateFunc: func(old, cur interface{}) {
			m, ok := cur.(*v1.ConfigMap)
			if !ok || m == nil {
				klog.Errorf("Expected v1.ConfigMap, item=%+v, typeIsOk=%v", cur, ok)
				return
			}

			if m.Namespace != UIDNamespace ||
				m.Name != UIDConfigMapName {
				return
			}

			if reflect.DeepEqual(old, cur) {
				return
			}

			klog.V(4).Infof("Observed updated configmap for clusteriD %v, %v; setting local values", m.Name, m.Data)
			g.ClusterID.update(m)
		},
	}

	listerWatcher := cache.NewListWatchFromClient(g.ClusterID.client.CoreV1().RESTClient(), "configmaps", UIDNamespace, fields.Everything())
	var controller cache.Controller
	g.ClusterID.store, controller = cache.NewInformer(newSingleObjectListerWatcher(listerWatcher, UIDConfigMapName), &v1.ConfigMap{}, updateFuncFrequency, mapEventHandler)

	controller.Run(stop)
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

// GetFederationID returns the id which could represent the entire Federation
// or just the cluster if not federated.
func (ci *ClusterID) GetFederationID() (string, bool, error) {
	if err := ci.getOrInitialize(); err != nil {
		return "", false, err
	}

	ci.idLock.RLock()
	defer ci.idLock.RUnlock()
	if ci.clusterID == nil {
		return "", false, errors.New("could not retrieve cluster id")
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
		return errors.New("Cloud.ClusterID is not ready. Call Initialize() before using")
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
	newID, err := makeUID()
	if err != nil {
		return err
	}

	klog.V(4).Infof("Creating clusteriD: %v", newID)
	cfg := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      UIDConfigMapName,
			Namespace: UIDNamespace,
		},
	}
	cfg.Data = map[string]string{
		UIDCluster:  newID,
		UIDProvider: newID,
	}

	if _, err := ci.client.CoreV1().ConfigMaps(UIDNamespace).Create(cfg); err != nil {
		klog.Errorf("GCE cloud provider failed to create %v config map to store cluster id: %v", ci.cfgMapKey, err)
		return err
	}

	klog.V(2).Infof("Created a config map containing clusteriD: %v", newID)
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
		klog.Error(err)
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
	if provID, exists := m.Data[UIDProvider]; exists {
		ci.providerID = &provID
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
