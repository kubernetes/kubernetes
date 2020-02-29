/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"context"
	"net/http"
	"net/http/httptest"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog"
	pvutil "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	"k8s.io/kubernetes/test/integration/framework"
)

// ShutdownFunc represents the function handle to be called, typically in a defer handler, to shutdown a running module
type ShutdownFunc func()

// StartApiserver starts a local API server for testing and returns the handle to the URL and the shutdown function to stop it.
func StartApiserver() (string, ShutdownFunc) {
	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	_, _, closeFn := framework.RunAMasterUsingServer(framework.NewIntegrationTestMasterConfig(), s, h)

	shutdownFunc := func() {
		klog.Infof("destroying API server")
		closeFn()
		s.Close()
		klog.Infof("destroyed API server")
	}
	return s.URL, shutdownFunc
}

// StartScheduler configures and starts a scheduler given a handle to the clientSet interface
// and event broadcaster. It returns the running scheduler and the shutdown function to stop it.
func StartScheduler(clientSet clientset.Interface) (*scheduler.Scheduler, coreinformers.PodInformer, ShutdownFunc) {
	ctx, cancel := context.WithCancel(context.Background())

	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)
	podInformer := informerFactory.Core().V1().Pods()
	evtBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{
		Interface: clientSet.EventsV1beta1().Events("")})

	evtBroadcaster.StartRecordingToSink(ctx.Done())

	sched, err := scheduler.New(
		clientSet,
		informerFactory,
		podInformer,
		profile.NewRecorderFactory(evtBroadcaster),
		ctx.Done())
	if err != nil {
		klog.Fatalf("Error creating scheduler: %v", err)
	}

	informerFactory.Start(ctx.Done())
	go sched.Run(ctx)

	shutdownFunc := func() {
		klog.Infof("destroying scheduler")
		cancel()
		klog.Infof("destroyed scheduler")
	}
	return sched, podInformer, shutdownFunc
}

// StartFakePVController is a simplified pv controller logic that sets PVC VolumeName and annotation for each PV binding.
// TODO(mborsz): Use a real PV controller here.
func StartFakePVController(clientSet clientset.Interface) ShutdownFunc {
	ctx, cancel := context.WithCancel(context.Background())

	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)
	pvInformer := informerFactory.Core().V1().PersistentVolumes()

	syncPV := func(obj *v1.PersistentVolume) {
		if obj.Spec.ClaimRef != nil {
			claimRef := obj.Spec.ClaimRef
			pvc, err := clientSet.CoreV1().PersistentVolumeClaims(claimRef.Namespace).Get(ctx, claimRef.Name, metav1.GetOptions{})
			if err != nil {
				klog.Errorf("error while getting %v/%v: %v", claimRef.Namespace, claimRef.Name, err)
				return
			}

			if pvc.Spec.VolumeName == "" {
				pvc.Spec.VolumeName = obj.Name
				metav1.SetMetaDataAnnotation(&pvc.ObjectMeta, pvutil.AnnBindCompleted, "yes")
				_, err := clientSet.CoreV1().PersistentVolumeClaims(claimRef.Namespace).Update(ctx, pvc, metav1.UpdateOptions{})
				if err != nil {
					klog.Errorf("error while getting %v/%v: %v", claimRef.Namespace, claimRef.Name, err)
					return
				}
			}
		}
	}

	pvInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			syncPV(obj.(*v1.PersistentVolume))
		},
		UpdateFunc: func(_, obj interface{}) {
			syncPV(obj.(*v1.PersistentVolume))
		},
	})

	informerFactory.Start(ctx.Done())
	return ShutdownFunc(cancel)
}
