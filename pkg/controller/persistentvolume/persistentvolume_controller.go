/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package persistentvolume

import (
	"time"

	"fmt"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/watch"
)

// PersistentVolumeController is a controller that watches for PersistentVolumes that are released from their claims.
// This controller will Recycle those volumes whose reclaim policy is set to PersistentVolumeReclaimRecycle and make them
// available again for a new claim.
type PersistentVolumeController struct {
	volumeController *framework.Controller
	claimController  *framework.Controller
	volumeStore      cache.Store
	claimStore       cache.Store
	stopChannel      chan struct{}
	kubeClient       clientset.Interface
	syncPeriod       time.Duration

	// all tasks execute within a context and access what they require through a host
	taskHost *taskHost
}

// NewPersistentVolumeController creates a new PersistentVolumeController
func NewPersistentVolumeController(kubeClient clientset.Interface, syncPeriod time.Duration, plugins []volume.VolumePlugin, cloud cloudprovider.Interface) (*PersistentVolumeController, error) {
	ctrl := &PersistentVolumeController{
		syncPeriod: syncPeriod,
	}

	taskClient := newTaskClient(kubeClient)
	taskHost, err := newTaskHost(taskClient, NewPersistentVolumeOrderedIndex(), ctrl.claimStore, workqueue.New(), plugins, cloud)
	if err != nil {
		return nil, fmt.Errorf("error creating taskHost: %v", err)
	}
	ctrl.taskHost = taskHost

	// this controller uses a custom store for indexing available PVs. NewInformer doesn't take a Store argument
	// for the watch to keep in sync.  We keep the custom store in sync in the handler funcs.
	_, ctrl.volumeController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return taskClient.ListPersistentVolumes(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return taskClient.WatchPersistentVolumes(options)
			},
		},
		&api.PersistentVolume{},
		// default sync is 10m
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				if pv, ok := obj.(*api.PersistentVolume); ok {
					ctrl.taskHost.volumeStore.Add(pv)
					ctrl.enqueueVolume(pv)
				}
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				if pv, ok := newObj.(*api.PersistentVolume); ok {
					ctrl.taskHost.volumeStore.Update(pv)
					ctrl.enqueueVolume(pv)
				}
			},
			DeleteFunc: func(obj interface{}) {
				if unk, ok := obj.(*cache.DeletedFinalStateUnknown); ok {
					// object may be stale but that's ok to keep a store in sync
					obj = unk.Obj
				}
				if pv, ok := obj.(*api.PersistentVolume); ok {
					ctrl.taskHost.volumeStore.Delete(pv)
				}
			},
		})

	taskHost.claimStore, ctrl.claimController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return taskHost.client.ListPersistentVolumeClaims(api.NamespaceAll, options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return taskHost.client.WatchPersistentVolumeClaims(api.NamespaceAll, options)
			},
		},
		&api.PersistentVolumeClaim{},
		// default sync is 10m
		syncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				if pvc, ok := obj.(*api.PersistentVolumeClaim); ok {
					ctrl.enqueueClaim(pvc)
				}
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				if pvc, ok := newObj.(*api.PersistentVolumeClaim); ok {
					ctrl.enqueueClaim(pvc)
				}
			},
		})

	return ctrl, nil
}

func (ctrl *PersistentVolumeController) enqueueVolume(pv *api.PersistentVolume) error {
	// might be *api.PVC or DeletionFinalStateUnknown
	key, err := controller.KeyFunc(pv)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", pv, err)
		return err
	}

	// abstract this into a queue/worker thing for parallelism
	// lock by pv name to eliminate concurrent concerns
	taskContext := newTaskContext(ctrl.taskHost)
	if err := taskContext.syncVolume(key); err != nil {
		glog.Errorf("PersistentVolume[%s] error syncing: %v", key, err)
		return err
	}

	return nil
}

func (ctrl *PersistentVolumeController) enqueueClaim(pvc *api.PersistentVolumeClaim) error {
	// might be *api.PVC or DeletionFinalStateUnknown
	key, err := controller.KeyFunc(pvc)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", pvc, err)
		return err
	}

	// abstract this into a queue/worker thing for parallelism
	// lock by pv name to eliminate concurrent concerns
	taskContext := newTaskContext(ctrl.taskHost)
	if err := taskContext.syncClaim(key); err != nil {
		glog.Errorf("PersistentVolumeClaim[%s] error syncing: %v", key, err)
		return err
	}

	return nil
}

//
//// worker runs a worker thread that just dequeues items, processes them, and marks them done.
//// It enforces that syncClaim is never invoked concurrently with the same key.
//func (ctrl *PersistentVolumeController) worker(queueName string, queue *workqueue.Type, handleFunc func(obj interface{}) error) {
//	for {
//		func() {
//			glog.V(5).Infof("Waiting on queue %s with %d depth", queueName, queue.Len())
//			key, quit := queue.Get()
//			if quit {
//				return
//			}
//			defer queue.Done(key)
//			glog.V(5).Infof("Invoking handler %#v", handleFunc)
//			err := handleFunc(key.(string))
//			if err != nil {
//				glog.Errorf("PersistentVolume[%s] error syncing: %v", err)
//			}
//		}()
//	}
//}

// Run starts this recycler's control loops
func (ctrl *PersistentVolumeController) Run(workers int, stopCh <-chan struct{}) {
	glog.V(5).Infof("Starting PersistentVolumeController\n")
	go ctrl.volumeController.Run(stopCh)
	go ctrl.claimController.Run(stopCh)
	//	for i := 0; i < workers; i++ {
	//		go wait.Until(func() { ctrl.worker("heartbeat", ctrl.volumeQueue, ctrl.syncVolume) }, time.Second, stopCh)
	//	}
}

// Stop gracefully shuts down this binder
func (recycler *PersistentVolumeController) Stop() {
	glog.V(5).Infof("Stopping PersistentVolumeController\n")
	if recycler.stopChannel != nil {
		close(recycler.stopChannel)
		recycler.stopChannel = nil
	}
}
