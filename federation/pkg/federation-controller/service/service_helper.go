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
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	cache "k8s.io/client-go/tools/cache"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/controller"

	"reflect"
	"sort"

	"github.com/golang/glog"
)

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (sc *ServiceController) clusterServiceWorker() {
	// process all pending events in serviceWorkerDoneChan
ForLoop:
	for {
		select {
		case clusterName := <-sc.serviceWorkerDoneChan:
			sc.serviceWorkerMap[clusterName] = false
		default:
			// non-blocking, comes here if all existing events are processed
			break ForLoop
		}
	}

	for clusterName, cache := range sc.clusterCache.clientMap {
		workerExist, found := sc.serviceWorkerMap[clusterName]
		if found && workerExist {
			continue
		}

		// create a worker only if the previous worker has finished and gone out of scope
		go func(cache *clusterCache, clusterName string) {
			fedClient := sc.federationClient
			for {
				func() {
					key, quit := cache.serviceQueue.Get()
					if quit {
						// send signal that current worker has finished tasks and is going out of scope
						sc.serviceWorkerDoneChan <- clusterName
						return
					}
					defer cache.serviceQueue.Done(key)
					err := sc.clusterCache.syncService(key.(string), clusterName, cache, sc.serviceCache, fedClient, sc)
					if err != nil {
						glog.Errorf("Failed to sync service: %+v", err)
					}
				}()
			}
		}(cache, clusterName)
		sc.serviceWorkerMap[clusterName] = true
	}
}

// Whenever there is change on service, the federation service should be updated
func (cc *clusterClientCache) syncService(key, clusterName string, clusterCache *clusterCache, serviceCache *serviceCache, fedClient fedclientset.Interface, sc *ServiceController) error {
	// obj holds the latest service info from apiserver, return if there is no federation cache for the service
	cachedService, ok := serviceCache.get(key)
	if !ok {
		// if serviceCache does not exists, that means the service is not created by federation, we should skip it
		return nil
	}
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		glog.Errorf("Did not successfully get %v from store: %v, will retry later", key, err)
		clusterCache.serviceQueue.Add(key)
		return err
	}
	var needUpdate, isDeletion bool
	service, err := clusterCache.serviceStore.Services(namespace).Get(name)
	switch {
	case errors.IsNotFound(err):
		glog.Infof("Can not get service %v for cluster %s from serviceStore", key, clusterName)
		needUpdate = cc.processServiceDeletion(cachedService, clusterName)
		isDeletion = true
	case err != nil:
		glog.Errorf("Did not successfully get %v from store: %v, will retry later", key, err)
		clusterCache.serviceQueue.Add(key)
		return err
	default:
		glog.V(4).Infof("Found service for federation service %s/%s from cluster %s", service.Namespace, service.Name, clusterName)
		needUpdate = cc.processServiceUpdate(cachedService, service, clusterName)
	}

	if needUpdate {
		for i := 0; i < clientRetryCount; i++ {
			err := sc.ensureDnsRecords(clusterName, cachedService)
			if err == nil {
				break
			}
			glog.V(4).Infof("Error ensuring DNS Records for service %s on cluster %s: %v", key, clusterName, err)
			time.Sleep(cachedService.nextDNSUpdateDelay())
			clusterCache.serviceQueue.Add(key)
			// did not retry here as we still want to persist federation apiserver even ensure dns records fails
		}
		err := cc.persistFedServiceUpdate(cachedService, fedClient)
		if err == nil {
			cachedService.appliedState = cachedService.lastState
			cachedService.resetFedUpdateDelay()
		} else {
			if err != nil {
				glog.Errorf("Failed to sync service: %+v, put back to service queue", err)
				clusterCache.serviceQueue.Add(key)
			}
		}
	}
	if isDeletion {
		// cachedService is not reliable here as
		// deleting cache is the last step of federation service deletion
		_, err := fedClient.Core().Services(cachedService.lastState.Namespace).Get(cachedService.lastState.Name, metav1.GetOptions{})
		// rebuild service if federation service still exists
		if err == nil || !errors.IsNotFound(err) {
			return sc.ensureClusterService(cachedService, clusterName, cachedService.appliedState, clusterCache.clientset)
		}
	}
	return nil
}

// processServiceDeletion is triggered when a service is delete from underlying k8s cluster
// the deletion function will wip out the cached ingress info of the service from federation service ingress
// the function returns a bool to indicate if actual update happened on federation service cache
// and if the federation service cache is updated, the updated info should be post to federation apiserver
func (cc *clusterClientCache) processServiceDeletion(cachedService *cachedService, clusterName string) bool {
	cachedService.rwlock.Lock()
	defer cachedService.rwlock.Unlock()
	cachedStatus, ok := cachedService.serviceStatusMap[clusterName]
	// cached status found, remove ingress info from federation service cache
	if ok {
		cachedFedServiceStatus := cachedService.lastState.Status.LoadBalancer
		removeIndexes := []int{}
		for i, fed := range cachedFedServiceStatus.Ingress {
			for _, new := range cachedStatus.Ingress {
				// remove if same ingress record found
				if new.IP == fed.IP && new.Hostname == fed.Hostname {
					removeIndexes = append(removeIndexes, i)
				}
			}
		}
		sort.Ints(removeIndexes)
		for i := len(removeIndexes) - 1; i >= 0; i-- {
			cachedFedServiceStatus.Ingress = append(cachedFedServiceStatus.Ingress[:removeIndexes[i]], cachedFedServiceStatus.Ingress[removeIndexes[i]+1:]...)
			glog.V(4).Infof("Remove old ingress %d for service %s/%s", removeIndexes[i], cachedService.lastState.Namespace, cachedService.lastState.Name)
		}
		delete(cachedService.serviceStatusMap, clusterName)
		delete(cachedService.endpointMap, clusterName)
		cachedService.lastState.Status.LoadBalancer = cachedFedServiceStatus
		return true
	} else {
		glog.V(4).Infof("Service removal %s/%s from cluster %s observed.", cachedService.lastState.Namespace, cachedService.lastState.Name, clusterName)
	}
	return false
}

// processServiceUpdate Update ingress info when service updated
// the function returns a bool to indicate if actual update happened on federation service cache
// and if the federation service cache is updated, the updated info should be post to federation apiserver
func (cc *clusterClientCache) processServiceUpdate(cachedService *cachedService, service *v1.Service, clusterName string) bool {
	glog.V(4).Infof("Processing service update for %s/%s, cluster %s", service.Namespace, service.Name, clusterName)
	cachedService.rwlock.Lock()
	defer cachedService.rwlock.Unlock()
	var needUpdate bool
	newServiceLB := service.Status.LoadBalancer
	cachedFedServiceStatus := cachedService.lastState.Status.LoadBalancer
	if len(newServiceLB.Ingress) == 0 {
		// not yet get LB IP
		return false
	}

	cachedStatus, ok := cachedService.serviceStatusMap[clusterName]
	if ok {
		if reflect.DeepEqual(cachedStatus, newServiceLB) {
			glog.V(4).Infof("Same ingress info observed for service %s/%s: %+v ", service.Namespace, service.Name, cachedStatus.Ingress)
		} else {
			glog.V(4).Infof("Ingress info was changed for service %s/%s: cache: %+v, new: %+v ",
				service.Namespace, service.Name, cachedStatus.Ingress, newServiceLB)
			needUpdate = true
		}
	} else {
		glog.V(4).Infof("Cached service status was not found for %s/%s, cluster %s, building one", service.Namespace, service.Name, clusterName)

		// cache is not always reliable(cache will be cleaned when service controller restart)
		// two cases will run into this branch:
		// 1. new service loadbalancer info received -> no info in cache, and no in federation service
		// 2. service controller being restarted -> no info in cache, but it is in federation service

		// check if the lb info is already in federation service

		cachedService.serviceStatusMap[clusterName] = newServiceLB
		needUpdate = false
		// iterate service ingress info
		for _, new := range newServiceLB.Ingress {
			var found bool
			// if it is known by federation service
			for _, fed := range cachedFedServiceStatus.Ingress {
				if new.IP == fed.IP && new.Hostname == fed.Hostname {
					found = true
					break
				}
			}
			if !found {
				needUpdate = true
				break
			}
		}
	}

	if needUpdate {
		// new status = cached federation status - cached status + new status from k8s cluster

		removeIndexes := []int{}
		for i, fed := range cachedFedServiceStatus.Ingress {
			for _, new := range cachedStatus.Ingress {
				// remove if same ingress record found
				if new.IP == fed.IP && new.Hostname == fed.Hostname {
					removeIndexes = append(removeIndexes, i)
				}
			}
		}
		sort.Ints(removeIndexes)
		for i := len(removeIndexes) - 1; i >= 0; i-- {
			cachedFedServiceStatus.Ingress = append(cachedFedServiceStatus.Ingress[:removeIndexes[i]], cachedFedServiceStatus.Ingress[removeIndexes[i]+1:]...)
		}
		cachedFedServiceStatus.Ingress = append(cachedFedServiceStatus.Ingress, service.Status.LoadBalancer.Ingress...)
		cachedService.lastState.Status.LoadBalancer = cachedFedServiceStatus
		glog.V(4).Infof("Add new ingress info %+v for service %s/%s", service.Status.LoadBalancer, service.Namespace, service.Name)
	} else {
		glog.V(4).Infof("Same ingress info found for %s/%s, cluster %s", service.Namespace, service.Name, clusterName)
	}
	return needUpdate
}

func (cc *clusterClientCache) persistFedServiceUpdate(cachedService *cachedService, fedClient fedclientset.Interface) error {
	service := cachedService.lastState
	glog.V(5).Infof("Persist federation service status %s/%s", service.Namespace, service.Name)
	var err error
	for i := 0; i < clientRetryCount; i++ {
		_, err := fedClient.Core().Services(service.Namespace).Get(service.Name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			glog.Infof("Not persisting update to service '%s/%s' that no longer exists: %v",
				service.Namespace, service.Name, err)
			return nil
		}
		_, err = fedClient.Core().Services(service.Namespace).UpdateStatus(service)
		if err == nil {
			glog.V(2).Infof("Successfully update service %s/%s to federation apiserver", service.Namespace, service.Name)
			return nil
		}
		if errors.IsNotFound(err) {
			glog.Infof("Not persisting update to service '%s/%s' that no longer exists: %v",
				service.Namespace, service.Name, err)
			return nil
		}
		if errors.IsConflict(err) {
			glog.V(4).Infof("Not persisting update to service '%s/%s' that has been changed since we received it: %v",
				service.Namespace, service.Name, err)
			return err
		}
		time.Sleep(cachedService.nextFedUpdateDelay())
	}
	return err
}

// obj could be an *api.Service, or a DeletionFinalStateUnknown marker item.
func (cc *clusterClientCache) enqueueService(obj interface{}, clusterName string) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	_, ok := cc.clientMap[clusterName]
	if ok {
		cc.clientMap[clusterName].serviceQueue.Add(key)
	}
}
