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

// Package volume implements a controller to manage volume attach and detach
// operations.
package volume

import (
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/controller/framework/informers"
	"k8s.io/kubernetes/pkg/util/runtime"
)

// AttachDetachController defines the operations supported by this controller.
type AttachDetachController interface {
	Run(stopCh <-chan struct{})
}

type attachDetachController struct {
	// internalPodInformer is the shared pod informer used to fetch and store
	// pod objects from the API server. It is shared with other controllers and
	// therefore the pod objects in its store should be treated as immutable.
	internalPodInformer framework.SharedInformer

	// selfCreatedPodInformer is true if the internalPodInformer was created
	// during initialization, not passed in.
	selfCreatedPodInformer bool

	// internalNodeInformer is the shared node informer used to fetch and store
	// node objects from the API server. It is shared with other controllers
	// and therefore the node objects in its store should be treated as
	// immutable.
	internalNodeInformer framework.SharedInformer

	// selfCreatedNodeInformer is true if the internalNodeInformer was created
	// during initialization, not passed in.
	selfCreatedNodeInformer bool
}

// NewAttachDetachController returns a new instance of AttachDetachController.
func NewAttachDetachController(
	kubeClient internalclientset.Interface,
	podInformer framework.SharedInformer,
	nodeInformer framework.SharedInformer,
	resyncPeriod time.Duration) AttachDetachController {
	selfCreatedPodInformer := false
	selfCreatedNodeInformer := false
	if podInformer == nil {
		podInformer = informers.CreateSharedPodInformer(kubeClient, resyncPeriod)
		selfCreatedPodInformer = true
	}
	if nodeInformer == nil {
		nodeInformer = informers.CreateSharedNodeIndexInformer(kubeClient, resyncPeriod)
		selfCreatedNodeInformer = true
	}

	adc := &attachDetachController{
		internalPodInformer:     podInformer,
		selfCreatedPodInformer:  selfCreatedPodInformer,
		internalNodeInformer:    nodeInformer,
		selfCreatedNodeInformer: selfCreatedNodeInformer,
	}

	podInformer.AddEventHandler(framework.ResourceEventHandlerFuncs{
		AddFunc:    adc.podAdd,
		UpdateFunc: adc.podUpdate,
		DeleteFunc: adc.podDelete,
	})

	nodeInformer.AddEventHandler(framework.ResourceEventHandlerFuncs{
		AddFunc:    adc.nodeAdd,
		UpdateFunc: adc.nodeUpdate,
		DeleteFunc: adc.nodeDelete,
	})

	return adc
}

func (adc *attachDetachController) Run(stopCh <-chan struct{}) {
	defer runtime.HandleCrash()
	glog.Infof("Starting Attach Detach Controller")

	// Start self-created shared informers
	if adc.selfCreatedPodInformer {
		go adc.internalPodInformer.Run(stopCh)
	}

	if adc.selfCreatedNodeInformer {
		go adc.internalNodeInformer.Run(stopCh)
	}

	<-stopCh
	glog.Infof("Shutting down Attach Detach Controller")
}

func (adc *attachDetachController) podAdd(obj interface{}) {
	// No op for now
}

func (adc *attachDetachController) podUpdate(oldObj, newObj interface{}) {
	// No op for now
}

func (adc *attachDetachController) podDelete(obj interface{}) {
	// No op for now
}

func (adc *attachDetachController) nodeAdd(obj interface{}) {
	// No op for now
}

func (adc *attachDetachController) nodeUpdate(oldObj, newObj interface{}) {
	// No op for now
}

func (adc *attachDetachController) nodeDelete(obj interface{}) {
	// No op for now
}
