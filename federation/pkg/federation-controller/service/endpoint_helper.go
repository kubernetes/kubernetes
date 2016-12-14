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

package service

import (
	"fmt"
	"time"

	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	cache "k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (sc *ServiceController) clusterEndpointWorker() {
	// process all pending events in endpointWorkerDoneChan
ForLoop:
	for {
		select {
		case clusterName := <-sc.endpointWorkerDoneChan:
			sc.endpointWorkerMap[clusterName] = false
		default:
			// non-blocking, comes here if all existing events are processed
			break ForLoop
		}
	}

	for clusterName, cache := range sc.clusterCache.clientMap {
		workerExist, found := sc.endpointWorkerMap[clusterName]
		if found && workerExist {
			continue
		}

		// create a worker only if the previous worker has finished and gone out of scope
		go func(cache *clusterCache, clusterName string) {
			fedClient := sc.federationClient
			for {
				func() {
					key, quit := cache.endpointQueue.Get()
					// update endpoint cache
					if quit {
						// send signal that current worker has finished tasks and is going out of scope
						sc.endpointWorkerDoneChan <- clusterName
						return
					}
					defer cache.endpointQueue.Done(key)
					err := sc.clusterCache.syncEndpoint(key.(string), clusterName, cache, sc.serviceCache, fedClient, sc)
					if err != nil {
						glog.V(2).Infof("Failed to sync endpoint: %+v", err)
					}
				}()
			}
		}(cache, clusterName)
		sc.endpointWorkerMap[clusterName] = true
	}
}

// Whenever there is change on endpoint, the federation service should be updated
// key is the namespaced name of endpoint
func (cc *clusterClientCache) syncEndpoint(key, clusterName string, clusterCache *clusterCache, serviceCache *serviceCache, fedClient fedclientset.Interface, serviceController *ServiceController) error {
	cachedService, ok := serviceCache.get(key)
	if !ok {
		// here we filtered all non-federation services
		return nil
	}
	endpointInterface, exists, err := clusterCache.endpointStore.GetByKey(key)
	if err != nil {
		glog.Errorf("Did not successfully get %v from store: %v, will retry later", key, err)
		clusterCache.endpointQueue.Add(key)
		return err
	}
	if exists {
		endpoint, ok := endpointInterface.(*v1.Endpoints)
		if ok {
			glog.V(4).Infof("Found endpoint for federation service %s/%s from cluster %s", endpoint.Namespace, endpoint.Name, clusterName)
			err = cc.processEndpointUpdate(cachedService, endpoint, clusterName, serviceController)
		} else {
			_, ok := endpointInterface.(cache.DeletedFinalStateUnknown)
			if !ok {
				return fmt.Errorf("Object contained wasn't a service or a deleted key: %+v", endpointInterface)
			}
			glog.Infof("Found tombstone for %v", key)
			err = cc.processEndpointDeletion(cachedService, clusterName, serviceController)
		}
	} else {
		// service absence in store means watcher caught the deletion, ensure LB info is cleaned
		glog.Infof("Can not get endpoint %v for cluster %s from endpointStore", key, clusterName)
		err = cc.processEndpointDeletion(cachedService, clusterName, serviceController)
	}
	if err != nil {
		glog.Errorf("Failed to sync service: %+v, put back to service queue", err)
		clusterCache.endpointQueue.Add(key)
	}
	cachedService.resetDNSUpdateDelay()
	return nil
}

func (cc *clusterClientCache) processEndpointDeletion(cachedService *cachedService, clusterName string, serviceController *ServiceController) error {
	glog.V(4).Infof("Processing endpoint deletion for %s/%s, cluster %s", cachedService.lastState.Namespace, cachedService.lastState.Name, clusterName)
	var err error
	cachedService.rwlock.Lock()
	defer cachedService.rwlock.Unlock()
	_, ok := cachedService.endpointMap[clusterName]
	// TODO remove ok checking? if service controller is restarted, then endpointMap for the cluster does not exist
	// need to query dns info from dnsprovider and make sure of if deletion is needed
	if ok {
		// endpoints lost, clean dns record
		glog.V(4).Infof("Cached endpoint was found for %s/%s, cluster %s, removing", cachedService.lastState.Namespace, cachedService.lastState.Name, clusterName)
		delete(cachedService.endpointMap, clusterName)
		for i := 0; i < clientRetryCount; i++ {
			err := serviceController.ensureDnsRecords(clusterName, cachedService)
			if err == nil {
				return nil
			}
			glog.V(4).Infof("Error ensuring DNS Records: %v", err)
			time.Sleep(cachedService.nextDNSUpdateDelay())
		}
	}
	return err
}

// Update dns info when endpoint update event received
// We do not care about the endpoint info, what we need to make sure here is len(endpoints.subsets)>0
func (cc *clusterClientCache) processEndpointUpdate(cachedService *cachedService, endpoint *v1.Endpoints, clusterName string, serviceController *ServiceController) error {
	glog.V(4).Infof("Processing endpoint update for %s/%s, cluster %s", endpoint.Namespace, endpoint.Name, clusterName)
	var err error
	cachedService.rwlock.Lock()
	var reachable bool
	defer cachedService.rwlock.Unlock()
	_, ok := cachedService.endpointMap[clusterName]
	if !ok {
		for _, subset := range endpoint.Subsets {
			if len(subset.Addresses) > 0 {
				reachable = true
				break
			}
		}
		if reachable {
			// first time get endpoints, update dns record
			glog.V(4).Infof("Reachable endpoint was found for %s/%s, cluster %s, building endpointMap", endpoint.Namespace, endpoint.Name, clusterName)
			cachedService.endpointMap[clusterName] = 1
			for i := 0; i < clientRetryCount; i++ {
				err := serviceController.ensureDnsRecords(clusterName, cachedService)
				if err == nil {
					return nil
				}
				glog.V(4).Infof("Error ensuring DNS Records: %v", err)
				time.Sleep(cachedService.nextDNSUpdateDelay())
			}
			return err
		}
	} else {
		for _, subset := range endpoint.Subsets {
			if len(subset.Addresses) > 0 {
				reachable = true
				break
			}
		}
		if !reachable {
			// first time get endpoints, update dns record
			glog.V(4).Infof("Reachable endpoint was lost for %s/%s, cluster %s, deleting endpointMap", endpoint.Namespace, endpoint.Name, clusterName)
			delete(cachedService.endpointMap, clusterName)
			for i := 0; i < clientRetryCount; i++ {
				err := serviceController.ensureDnsRecords(clusterName, cachedService)
				if err == nil {
					return nil
				}
				glog.V(4).Infof("Error ensuring DNS Records: %v", err)
				time.Sleep(cachedService.nextDNSUpdateDelay())
			}
			return err
		}
	}
	return nil
}

// obj could be an *api.Endpoints, or a DeletionFinalStateUnknown marker item.
func (cc *clusterClientCache) enqueueEndpoint(obj interface{}, clusterName string) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	_, ok := cc.clientMap[clusterName]
	if ok {
		cc.clientMap[clusterName].endpointQueue.Add(key)
	}
}
