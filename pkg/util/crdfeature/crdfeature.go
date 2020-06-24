/*
Copyright 2020 The Kubernetes Authors.

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

// package crdfeature contains tools for enabling and disabling features based on
// the presence of a required CRD.
package crdfeature

import (
	"fmt"
	"sync"
	"time"

	crdv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
)

func toCRD(obj interface{}) (*crdv1.CustomResourceDefinition, error) {
	resource, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return nil, fmt.Errorf("unexpected object type: %v", obj)
	}
	var crd crdv1.CustomResourceDefinition
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(resource.UnstructuredContent(), &crd); err != nil {
		return nil, err
	}
	return &crd, nil
}

type Feature interface {
	Enable()
	Disable()
}

type funcFeature struct {
	enable  func()
	disable func()
}

func (f *funcFeature) Enable() {
	f.enable()
}

func (f *funcFeature) Disable() {
	f.disable()
}

// WithFuncs creates a new Feature that is enabled and disabled
// by calling the supplied functions.
func WithFuncs(enable, disable func()) Feature {
	return &funcFeature{
		enable:  enable,
		disable: disable,
	}
}

type chanFeature struct {
	stopCh chan struct{}
	enable func(<-chan struct{})
}

func (f *chanFeature) Enable() {
	f.stopCh = make(chan struct{})
	f.enable(f.stopCh)
}

func (f *chanFeature) Disable() {
	stopCh := f.stopCh
	f.stopCh = nil
	close(stopCh)
}

// WithChan creates a new Feature that is enabled by calling the supplied
// function with a channel. The channel will be closed when the feature is
// disabled.
func WithChan(enable func(<-chan struct{})) Feature {
	return &chanFeature{enable: enable}
}

type Watcher struct {
	sync.Mutex
	gvr             schema.GroupVersionResource
	feature         Feature
	informerFactory dynamicinformer.DynamicSharedInformerFactory
	enabled         bool
}

func NewWatcher(client dynamic.Interface, gvr schema.GroupVersionResource, resyncPeriod time.Duration, feature Feature) *Watcher {
	watcher := Watcher{
		gvr:     gvr,
		feature: feature,
		informerFactory: dynamicinformer.NewFilteredDynamicSharedInformerFactory(client, resyncPeriod, metav1.NamespaceAll,
			func(options *metav1.ListOptions) {
				options.FieldSelector = fields.OneTermEqualSelector("metadata.name", gvr.GroupResource().String()).String()
			}),
	}

	watcher.informerFactory.ForResource(crdv1.SchemeGroupVersion.WithResource("customresourcedefinitions")).Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    watcher.handleAdd,
			UpdateFunc: watcher.handleUpdate,
			DeleteFunc: watcher.handleDelete,
		},
		resyncPeriod,
	)
	return &watcher
}

func (w *Watcher) Start(stopCh <-chan struct{}) {
	w.informerFactory.Start(stopCh)
}

func (w *Watcher) enableFeature() {
	w.Lock()
	defer w.Unlock()
	if w.enabled {
		return
	}
	w.enabled = true
	klog.Infof("Enabling feature for %v", w.gvr)
	w.feature.Enable()
	klog.V(2).Infof("Enabled feature for%v", w.gvr)
}

func (w *Watcher) disableFeature() {
	w.Lock()
	defer w.Unlock()
	if !w.enabled {
		return
	}
	w.enabled = false
	klog.V(2).Infof("Disabling feature for %v", w.gvr)
	w.feature.Disable()
	klog.Infof("Disabled feature for %v", w.gvr)
}

func (w *Watcher) crdMatchesGVR(crd *crdv1.CustomResourceDefinition) bool {
	if crd.Name != w.gvr.GroupResource().String() {
		return false
	}
	for _, version := range crd.Spec.Versions {
		if version.Name == w.gvr.Version && version.Served {
			return true
		}
	}
	return false
}

func (w *Watcher) handleAdd(obj interface{}) {
	crd, err := toCRD(obj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	if !w.crdMatchesGVR(crd) {
		utilruntime.HandleError(fmt.Errorf("expected crd version not supported: %v", w.gvr))
		return
	}
	w.enableFeature()
}

func (w *Watcher) handleUpdate(oldObj, newObj interface{}) {
	crd, err := toCRD(newObj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}

	if w.crdMatchesGVR(crd) {
		w.enableFeature()
	} else {
		utilruntime.HandleError(fmt.Errorf("expected crd version not supported: %v", w.gvr))
		w.disableFeature()
		return
	}
}

func (w *Watcher) handleDelete(obj interface{}) {
	w.disableFeature()
}
