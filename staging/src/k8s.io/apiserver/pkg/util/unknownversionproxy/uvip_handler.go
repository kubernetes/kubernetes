/*
Copyright 2023 The Kubernetes Authors.

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

package unknownversionproxy

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"k8s.io/api/apiserverinternal/v1alpha1"
	schema "k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storageversion"
	apiserverinternallister "k8s.io/client-go/listers/apiserverinternal/v1alpha1"
	coordlisters "k8s.io/client-go/listers/coordination/v1"
	"k8s.io/client-go/tools/cache"
)

var (
	finishedSync atomic.Bool
)

type uvipHandler struct {
	name                string
	svLister            apiserverinternallister.StorageVersionLister
	leaseLister         coordlisters.LeaseLister
	svi                 cache.SharedIndexInformer
	leasei              cache.SharedIndexInformer
	svm                 storageversion.Manager
	proxyClientCertFile string
	proxyClientKeyFile  string
	peerCAFile          string
	peerBindAddress     string
	svMap               sync.Map
	lock                sync.RWMutex
}

func NewUVIPHandler(config UVIPConfig) *uvipHandler {
	h := &uvipHandler{
		name:                config.Name,
		svm:                 config.Svm,
		proxyClientCertFile: config.ProxyClientCertFile,
		proxyClientKeyFile:  config.ProxyClientKeyFile,
		peerCAFile:          config.PeerCAFile,
		peerBindAddress:     config.PeerBindAddress,
	}
	finishedSync.Store(false)
	svi := config.InformerFactory.Internal().V1alpha1().StorageVersions()
	leasei := config.InformerFactory.Coordination().V1().Leases()
	h.svi = svi.Informer()
	h.svLister = svi.Lister()
	h.leasei = leasei.Informer()
	h.leaseLister = leasei.Lister()
	h.svMap = sync.Map{}

	svi.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			h.addSV(obj)
		},
		UpdateFunc: func(oldObj, newObj interface{}) {
			h.updateSV(oldObj, newObj)
		},
		// This will enter the sync loop and no-op, because the deployment has been deleted from the store.
		DeleteFunc: func(obj interface{}) {
			h.deleteSV(obj)
		}})
	return h
}

func (h *uvipHandler) addSV(obj interface{}) {
	sv := obj.(*v1alpha1.StorageVersion)
	h.updateSVMap(nil, sv)
}

func (h *uvipHandler) updateSV(oldObj interface{}, newObj interface{}) {
	oldSV := oldObj.(*v1alpha1.StorageVersion)
	newSV := newObj.(*v1alpha1.StorageVersion)
	h.updateSVMap(oldSV, newSV)
}

func (h *uvipHandler) deleteSV(obj interface{}) {
	sv := obj.(*v1alpha1.StorageVersion)
	h.updateSVMap(sv, nil)
}

func (h *uvipHandler) updateSVMap(oldSV *v1alpha1.StorageVersion, newSV *v1alpha1.StorageVersion) {

	h.lock.Lock()
	defer h.lock.Unlock()

	if oldSV != nil {
		// delete old SV entries
		h.deleteSVFromMap(oldSV)
	}

	if newSV != nil {
		// add new SV entries
		h.addSVToMap(newSV)
	}
}

func (h *uvipHandler) deleteSVFromMap(sv *v1alpha1.StorageVersion) {
	splitInd := strings.LastIndex(sv.Name, ".")
	group := sv.Name[:splitInd]
	resource := sv.Name[splitInd+1:]
	gvr := schema.GroupVersionResource{Group: group, Resource: resource}
	for _, gr := range sv.Status.StorageVersions {
		for _, version := range gr.DecodableVersions {
			versionSplit := strings.Split(version, "/")
			if len(versionSplit) == 2 {
				version = versionSplit[1]
			}
			gvr.Version = version
			h.svMap.Delete(gvr)
		}
	}
}

func (h *uvipHandler) addSVToMap(sv *v1alpha1.StorageVersion) {
	splitInd := strings.LastIndex(sv.Name, ".")
	group := sv.Name[:splitInd]
	resource := sv.Name[splitInd+1:]
	gvr := schema.GroupVersionResource{Group: group, Resource: resource}
	for _, gr := range sv.Status.StorageVersions {
		for _, version := range gr.DecodableVersions {
			versionSplit := strings.Split(version, "/")
			if len(versionSplit) == 2 {
				version = versionSplit[1]
			}
			gvr.Version = version
			apiserversi, _ := h.svMap.LoadOrStore(gvr, &sync.Map{})
			apiservers := apiserversi.(*sync.Map)
			apiservers.Store(gr.APIServerID, true)
		}
	}
}

func (h *uvipHandler) findServiceableByServers(gvr schema.GroupVersionResource, localAPIServerId string) (serviceableByResponse, error) {

	h.lock.RLock()
	defer h.lock.RUnlock()
	apiserversi, ok := h.svMap.LoadOrStore(gvr, &sync.Map{})
	apiservers := apiserversi.(*sync.Map)

	if !ok {
		return serviceableByResponse{}, fmt.Errorf("Error retrieving StorageVersions for the GVR: %v", gvr)
	}

	response := serviceableByResponse{}
	var serviceableBy []string
	apiservers.Range(func(key, value interface{}) bool {
		apiserverKey := key.(string)
		if apiserverKey == localAPIServerId {
			response.locallyServiceable = true
			return false
		}
		serviceableBy = append(serviceableBy, apiserverKey)
		return true
	})
	response.serviceableBy = serviceableBy
	return response, nil
}

func (h *uvipHandler) HasFinishedSync() bool {
	return finishedSync.Load()
}

type serviceableByResponse struct {
	locallyServiceable bool
	serviceableBy      []string
}
