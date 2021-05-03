/*
Copyright 2021 The Kubernetes Authors.

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

package startupcrd

import (
	"context"
	k8sapierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/discovery/cached/memory"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
)

// Installer stores information needed to install CRDs at startup
type Installer struct {
	stopCh       <-chan struct{}
	clientConfig *restclient.Config
}

// NewInstaller returns a fresh copy of installer
func NewInstaller(stopCh <-chan struct{}, clientConfig *restclient.Config) (*Installer, error) {
	// do some checks here if needed

	return &Installer{
		stopCh:       stopCh,
		clientConfig: clientConfig,
	}, nil
}

// NewInstallerFromHookContext returns a fresh copy of installer from a PostStartHookContext
func NewInstallerFromHookContext(hookContext genericapiserver.PostStartHookContext) (*Installer, error) {
	return NewInstaller(hookContext.StopCh, hookContext.LoopbackClientConfig)
}

// Install ensures the specified CRDs are installed
func (i *Installer) Install(objs []*unstructured.Unstructured) error {
	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		select {
		case <-i.stopCh:
			cancel()
		}
	}()

	// wait for CRD type to become available
	if err := wait.PollImmediateUntil(100*time.Millisecond, func() (bool, error) {
		return i.isCRDResourceAvailable()
	}, i.stopCh); err != nil {
		return err
	}

	// install the objects one by one
	for _, obj := range objs {
		if err := i.installOne(ctx, obj); err != nil {
			return err
		}
	}

	return nil
}

func (i *Installer) isCRDResourceAvailable() (bool, error) {
	// get a discovery client
	dc, err := discovery.NewDiscoveryClientForConfig(i.clientConfig)
	if err != nil {
		return false, err
	}

	// fetch all the api resources
	apiResources, err := restmapper.GetAPIGroupResources(dc)
	if err != nil {
		return false, err
	}

	// find whether "apiextensions.k8s.io" exists in the list of api resources
	for _, apiGroupResource := range apiResources {
		if apiGroupResource.Group.Name == "apiextensions.k8s.io" {
			return true, nil
		}
	}

	return false, nil
}

func (i *Installer) installOne(ctx context.Context, obj *unstructured.Unstructured) error {
	dc, err := discovery.NewDiscoveryClientForConfig(i.clientConfig)
	if err != nil {
		return err
	}
	mapper := restmapper.NewDeferredDiscoveryRESTMapper(memory.NewMemCacheClient(dc))

	gvk := obj.GroupVersionKind()
	mapping, err := mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
	if err != nil {
		return err
	}

	dyn, err := dynamic.NewForConfig(i.clientConfig)
	if err != nil {
		return err
	}

	var dynRes dynamic.ResourceInterface
	if mapping.Scope.Name() == meta.RESTScopeNameNamespace {
		dynRes = dyn.Resource(mapping.Resource).Namespace(obj.GetNamespace())
	} else {
		dynRes = dyn.Resource(mapping.Resource)
	}

	// check if the CRD exists
	// if it doesn't, create the CRD
	// if it does, patch the existing CRD with the CRD read passed
	if err := i.getResource(ctx, dynRes, obj); err != nil {
		if k8sapierrors.IsNotFound(err) {
			// The resource does not exist on the cluster. Create it.
			if err := i.createResource(ctx, dynRes, obj); err != nil {
				return err
			}
			return nil
		}
		return err
	} else {
		// The resource already exists on the cluster. Patch it.
		if err := i.patchResource(ctx, dynRes, obj); err != nil {
			return err
		}
	}
	return nil
}

func (i *Installer) getResource(ctx context.Context, dynRes dynamic.ResourceInterface, obj *unstructured.Unstructured) error {
	_, err := dynRes.Get(ctx, obj.GetName(), metav1.GetOptions{})
	if err != nil {
		return err
	}

	return nil
}

func (i *Installer) createResource(ctx context.Context, dynRes dynamic.ResourceInterface, obj *unstructured.Unstructured) error {
	_, err := dynRes.Create(ctx, obj, metav1.CreateOptions{FieldManager: "awesome-client"})
	if err != nil {
		return err
	}

	return nil
}

func (i *Installer) patchResource(ctx context.Context, dynRes dynamic.ResourceInterface, obj *unstructured.Unstructured) error {
	data, err := obj.MarshalJSON()
	if err != nil {
		return err
	}

	_, err = dynRes.Patch(ctx, obj.GetName(), types.JSONPatchType, data, metav1.PatchOptions{FieldManager: "awesome-client"})
	if err != nil {
		return err
	}

	return nil
}
