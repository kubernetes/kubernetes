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

package ingress

import (
	"fmt"
	"time"

	federation_release_1_4 "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/pkg/api/errors"
	v1 "k8s.io/kubernetes/pkg/api/v1"
	cache "k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/controller"

	"reflect"
	"sort"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (i *IngressController) clusterIngressWorker() {
	fedClient := i.federationClient
	for clusterName, cache := range i.clusterCache.clientMap {
		go func(cache *clusterCache, clusterName string) {
			for {
				func() {
					key, quit := cache.ingressQueue.Get()
					defer cache.ingressQueue.Done(key)
					if quit {
						return
					}
					err := i.clusterCache.syncIngress(key.(string), clusterName, cache, i.ingressCache, fedClient, i)
					if err != nil {
						glog.Errorf("Failed to sync ingress: %+v", err)
					}
				}()
			}
		}(cache, clusterName)
	}
}

// Whenever there is change on ingress, the federation ingress should be updated
func (cc *clusterClientCache) syncIngress(key, clusterName string, clusterCache *clusterCache, ingressCache *ingressCache, fedClient federation_release_1_4.Interface, ic *IngressController) error {
	// obj holds the latest ingress info from apiserver, return if there is no federation cache for the ingress
	cachedIngress, ok := ingressCache.get(key)
	if !ok {
		// if ingressCache does not exists, that means the ingress is not created by federation, we should skip it
		return nil
	}
	ingressInterface, exists, err := clusterCache.ingressStore.GetByKey(key)
	if err != nil {
		glog.Infof("Did not successfully get %v from store: %v, will retry later", key, err)
		clusterCache.ingressQueue.Add(key)
		return err
	}
	var needUpdate, isDeletion bool
	if exists {
		ingress, ok := ingressInterface.(*extensions.Ingress)
		if ok {
			glog.V(4).Infof("Found ingress for federation ingress %s/%s from cluster %s", ingress.Namespace, ingress.Name, clusterName)
			needUpdate = cc.processIngressUpdate(cachedIngress, ingress, clusterName)
		} else {
			_, ok := ingressInterface.(cache.DeletedFinalStateUnknown)
			if !ok {
				return fmt.Errorf("Object contained wasn't a ingress or a deleted key: %+v", ingressInterface)
			}
			glog.Infof("Found tombstone for %v", key)
			needUpdate = cc.processIngressDeletion(cachedIngress, clusterName)
			isDeletion = true
		}
	} else {
		glog.Infof("Can not get ingress %v for cluster %s from ingressStore", key, clusterName)
		needUpdate = cc.processIngressDeletion(cachedIngress, clusterName)
		isDeletion = true
	}

	if needUpdate {
		for i := 0; i < clientRetryCount; i++ {
			err := ic.ensureDnsRecords(clusterName, cachedIngress)
			if err == nil {
				break
			}
			glog.V(4).Infof("Error ensuring DNS Records for ingress %s on cluster %s: %v", key, clusterName, err)
			time.Sleep(cachedIngress.nextDNSUpdateDelay())
			clusterCache.ingressQueue.Add(key)
			// did not retry here as we still want to persist federation apiserver even ensure dns records fails
		}
		err := cc.persistFedIngressUpdate(cachedIngress, fedClient)
		if err == nil {
			cachedIngress.appliedState = cachedIngress.lastState
			cachedIngress.resetFedUpdateDelay()
		} else {
			if err != nil {
				glog.Errorf("Failed to sync ingress: %+v, put back to ingress queue", err)
				clusterCache.ingressQueue.Add(key)
			}
		}
	}
	if isDeletion {
		// cachedIngress is not reliable here as
		// deleting cache is the last step of federation ingress deletion
		// TODO
		//_, err := fedClient.Extensions().Ingresses(cachedIngress.lastState.Namespace).Get(cachedIngress.lastState.Name)
		// rebuild ingress if federation ingress still exists
		//if err == nil || !errors.IsNotFound(err) {
		//	return ic.ensureClusterIngress(cachedIngress, clusterName, cachedIngress.appliedState, clusterCache.clientset)
		//}
	}
	return nil
}

// processIngressDeletion is triggered when a ingress is delete from underlying k8s cluster
// the deletion function will wip out the cached ingress info of the ingress from federation ingress ingress
// the function returns a bool to indicate if actual update happened on federation ingress cache
// and if the federation ingress cache is updated, the updated info should be post to federation apiserver
func (cc *clusterClientCache) processIngressDeletion(cachedIngress *cachedIngress, clusterName string) bool {
	cachedIngress.rwlock.Lock()
	defer cachedIngress.rwlock.Unlock()
	cachedStatus, ok := cachedIngress.ingressStatusMap[clusterName]
	// cached status found, remove ingress info from federation ingress cache
	if ok {
		cachedFedIngressStatus := cachedIngress.lastState.Status.LoadBalancer
		removeIndexes := []int{}
		for i, fed := range cachedFedIngressStatus.Ingress {
			for _, new := range cachedStatus.LoadBalancer {
				// remove if same ingress record found
				if new.IP == fed.IP && new.Hostname == fed.Hostname {
					removeIndexes = append(removeIndexes, i)
				}
			}
		}
		sort.Ints(removeIndexes)
		for i := len(removeIndexes) - 1; i >= 0; i-- {
			cachedFedIngressStatus.Ingress = append(cachedFedIngressStatus.Ingress[:removeIndexes[i]], cachedFedIngressStatus.Ingress[removeIndexes[i]+1:]...)
			glog.V(4).Infof("Remove old ingress %d for ingress %s/%s", removeIndexes[i], cachedIngress.lastState.Namespace, cachedIngress.lastState.Name)
		}
		delete(cachedIngress.ingressStatusMap, clusterName)
		delete(cachedIngress.endpointMap, clusterName)
		cachedIngress.lastState.Status.LoadBalancer = cachedFedIngressStatus
		return true
	} else {
		glog.V(4).Infof("Ingress removal %s/%s from cluster %s observed.", cachedIngress.lastState.Namespace, cachedIngress.lastState.Name, clusterName)
	}
	return false
}

// processIngressUpdate Update ingress info when ingress updated
// the function returns a bool to indicate if actual update happened on federation ingress cache
// and if the federation ingress cache is updated, the updated info should be post to federation apiserver
func (cc *clusterClientCache) processIngressUpdate(cachedIngress *cachedIngress, ingress *v1.Ingress, clusterName string) bool {
	glog.V(4).Infof("Processing ingress update for %s/%s, cluster %s", ingress.Namespace, ingress.Name, clusterName)
	cachedIngress.rwlock.Lock()
	defer cachedIngress.rwlock.Unlock()
	var needUpdate bool
	newIngressLB := ingress.Status.LoadBalancer
	cachedFedIngressStatus := cachedIngress.lastState.Status.LoadBalancer
	if len(newIngressLB.Ingress) == 0 {
		// not yet get LB IP
		return false
	}

	cachedStatus, ok := cachedIngress.ingressStatusMap[clusterName]
	if ok {
		if reflect.DeepEqual(cachedStatus, newIngressLB) {
			glog.V(4).Infof("Same ingress info observed for ingress %s/%s: %+v ", ingress.Namespace, ingress.Name, cachedStatus.Ingress)
		} else {
			glog.V(4).Infof("Ingress info was changed for ingress %s/%s: cache: %+v, new: %+v ",
				ingress.Namespace, ingress.Name, cachedStatus.Ingress, newIngressLB)
			needUpdate = true
		}
	} else {
		glog.V(4).Infof("Cached ingress status was not found for %s/%s, cluster %s, building one", ingress.Namespace, ingress.Name, clusterName)

		// cache is not always reliable(cache will be cleaned when ingress controller restart)
		// two cases will run into this branch:
		// 1. new ingress loadbalancer info received -> no info in cache, and no in federation ingress
		// 2. ingress controller being restarted -> no info in cache, but it is in federation ingress

		// check if the lb info is already in federation ingress

		cachedIngress.ingressStatusMap[clusterName] = newIngressLB
		needUpdate = false
		// iterate ingress ingress info
		for _, new := range newIngressLB.Ingress {
			var found bool
			// if it is known by federation ingress
			for _, fed := range cachedFedIngressStatus.Ingress {
				if new.IP == fed.IP && new.Hostname == fed.Hostname {
					found = true
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
		for i, fed := range cachedFedIngressStatus.Ingress {
			for _, new := range cachedStatus.Ingress {
				// remove if same ingress record found
				if new.IP == fed.IP && new.Hostname == fed.Hostname {
					removeIndexes = append(removeIndexes, i)
				}
			}
		}
		sort.Ints(removeIndexes)
		for i := len(removeIndexes) - 1; i >= 0; i-- {
			cachedFedIngressStatus.Ingress = append(cachedFedIngressStatus.Ingress[:removeIndexes[i]], cachedFedIngressStatus.Ingress[removeIndexes[i]+1:]...)
		}
		cachedFedIngressStatus.Ingress = append(cachedFedIngressStatus.Ingress, ingress.Status.LoadBalancer.Ingress...)
		cachedIngress.lastState.Status.LoadBalancer = cachedFedIngressStatus
		glog.V(4).Infof("Add new ingress info %+v for ingress %s/%s", ingress.Status.LoadBalancer, ingress.Namespace, ingress.Name)
	} else {
		glog.V(4).Infof("Same ingress info found for %s/%s, cluster %s", ingress.Namespace, ingress.Name, clusterName)
	}
	return needUpdate
}

func (cc *clusterClientCache) persistFedIngressUpdate(cachedIngress *cachedIngress, fedClient federation_release_1_4.Interface) error {
	ingress := cachedIngress.lastState
	glog.V(5).Infof("Persist federation ingress status %s/%s", ingress.Namespace, ingress.Name)
	var err error
	for i := 0; i < clientRetryCount; i++ {
		_, err := fedClient.Core().Ingresss(ingress.Namespace).Get(ingress.Name)
		if errors.IsNotFound(err) {
			glog.Infof("Not persisting update to ingress '%s/%s' that no longer exists: %v",
				ingress.Namespace, ingress.Name, err)
			return nil
		}
		_, err = fedClient.Core().Ingresss(ingress.Namespace).UpdateStatus(ingress)
		if err == nil {
			glog.V(2).Infof("Successfully update ingress %s/%s to federation apiserver", ingress.Namespace, ingress.Name)
			return nil
		}
		if errors.IsNotFound(err) {
			glog.Infof("Not persisting update to ingress '%s/%s' that no longer exists: %v",
				ingress.Namespace, ingress.Name, err)
			return nil
		}
		if errors.IsConflict(err) {
			glog.V(4).Infof("Not persisting update to ingress '%s/%s' that has been changed since we received it: %v",
				ingress.Namespace, ingress.Name, err)
			return err
		}
		time.Sleep(cachedIngress.nextFedUpdateDelay())
	}
	return err
}

// obj could be an *api.Ingress, or a DeletionFinalStateUnknown marker item.
func (cc *clusterClientCache) enqueueIngress(obj interface{}, clusterName string) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	_, ok := cc.clientMap[clusterName]
	if ok {
		cc.clientMap[clusterName].ingressQueue.Add(key)
	}
}
