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

package rest

import (
	"context"
	"fmt"
	"time"

	"k8s.io/klog/v2"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	schedulingclient "k8s.io/client-go/kubernetes/typed/scheduling/v1"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	schedulingapiv1 "k8s.io/kubernetes/pkg/apis/scheduling/v1"
	priorityclassstore "k8s.io/kubernetes/pkg/registry/scheduling/priorityclass/storage"
)

const PostStartHookName = "scheduling/bootstrap-system-priority-classes"

type RESTStorageProvider struct{}

var _ genericapiserver.PostStartHookProvider = RESTStorageProvider{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(scheduling.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)

	if storageMap, err := p.v1Storage(apiResourceConfigSource, restOptionsGetter); err != nil {
		return genericapiserver.APIGroupInfo{}, err
	} else if len(storageMap) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[schedulingapiv1.SchemeGroupVersion.Version] = storageMap
	}

	return apiGroupInfo, nil
}

func (p RESTStorageProvider) v1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}

	// priorityclasses
	if resource := "priorityclasses"; apiResourceConfigSource.ResourceEnabled(schedulingapiv1.SchemeGroupVersion.WithResource(resource)) {
		if priorityClassStorage, err := priorityclassstore.NewREST(restOptionsGetter); err != nil {
			return nil, err
		} else {
			storage[resource] = priorityClassStorage
		}
	}

	return storage, nil
}

func (p RESTStorageProvider) PostStartHook() (string, genericapiserver.PostStartHookFunc, error) {
	return PostStartHookName, AddSystemPriorityClasses(), nil
}

func AddSystemPriorityClasses() genericapiserver.PostStartHookFunc {
	return func(hookContext genericapiserver.PostStartHookContext) error {
		// Adding system priority classes is important. If they fail to add, many critical system
		// components may fail and cluster may break.
		err := wait.Poll(1*time.Second, 30*time.Second, func() (done bool, err error) {
			schedClientSet, err := schedulingclient.NewForConfig(hookContext.LoopbackClientConfig)
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("unable to initialize client: %v", err))
				return false, nil
			}

			for _, pc := range schedulingapiv1.SystemPriorityClasses() {
				_, err := schedClientSet.PriorityClasses().Get(context.TODO(), pc.Name, metav1.GetOptions{})
				if err != nil {
					if apierrors.IsNotFound(err) {
						_, err := schedClientSet.PriorityClasses().Create(context.TODO(), pc, metav1.CreateOptions{})
						if err == nil || apierrors.IsAlreadyExists(err) {
							klog.Infof("created PriorityClass %s with value %v", pc.Name, pc.Value)
							continue
						}
						// ServiceUnavailable error is returned when the API server is blocked by storage version updates
						if apierrors.IsServiceUnavailable(err) {
							klog.Infof("going to retry, unable to create PriorityClass %s: %v", pc.Name, err)
							return false, nil
						}
						return false, err
					} else {
						// Unable to get the priority class for reasons other than "not found".
						klog.Warningf("unable to get PriorityClass %v: %v. Retrying...", pc.Name, err)
						return false, nil
					}
				}
			}
			klog.Infof("all system priority classes are created successfully or already exist.")
			return true, nil
		})
		// if we're never able to make it through initialization, kill the API server.
		if err != nil {
			return fmt.Errorf("unable to add default system priority classes: %v", err)
		}
		return nil
	}
}

func (p RESTStorageProvider) GroupName() string {
	return scheduling.GroupName
}
