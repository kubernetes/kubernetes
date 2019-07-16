/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"time"

	flowcontrolv1alpha1 "k8s.io/api/flowcontrol/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	flowcontrolclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1alpha1"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
	"k8s.io/kubernetes/pkg/apis/flowcontrol/bootstrap"
	flowschemastore "k8s.io/kubernetes/pkg/registry/flowcontrol/flowschema/storage"
	prioritylevelconfigurationstore "k8s.io/kubernetes/pkg/registry/flowcontrol/prioritylevelconfiguration/storage"
)

const PostStartHookName = "apiserver/bootstrap-system-flowcontrol-configuration"

type RESTStorageProvider struct{}

var _ genericapiserver.PostStartHookProvider = RESTStorageProvider{}

func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(flowcontrol.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)

	if apiResourceConfigSource.VersionEnabled(flowcontrolv1alpha1.SchemeGroupVersion) {
		apiGroupInfo.VersionedResourcesStorageMap[flowcontrolv1alpha1.SchemeGroupVersion.Version] = p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter)
	}
	return apiGroupInfo, true
}

func (p RESTStorageProvider) v1alpha1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) map[string]rest.Storage {
	storage := map[string]rest.Storage{}

	// flow-schema
	flowSchemaStorage, flowSchemaStatusStorage := flowschemastore.NewREST(restOptionsGetter)
	storage["flowschemas"] = flowSchemaStorage
	storage["flowschemas/status"] = flowSchemaStatusStorage

	// priority-level-configuration
	priorityLevelConfigurationStorage, priorityLevelConfigurationStatusStorage := prioritylevelconfigurationstore.NewREST(restOptionsGetter)
	storage["prioritylevelconfigurations"] = priorityLevelConfigurationStorage
	storage["prioritylevelconfigurations/status"] = priorityLevelConfigurationStatusStorage

	return storage
}

func (p RESTStorageProvider) PostStartHook() (string, genericapiserver.PostStartHookFunc, error) {
	systemPreset := SystemPresetData{
		FlowSchemas:                 bootstrap.SystemFlowSchemas(),
		PriorityLevelConfigurations: bootstrap.SystemPriorityLevelConfigurations(),
	}
	// TODO: default flow-schemas and priority levels
	return PostStartHookName, systemPreset.EnsureSystemPresetConfiguration(), nil
}

type SystemPresetData struct {
	FlowSchemas                 []*flowcontrolv1alpha1.FlowSchema
	PriorityLevelConfigurations []*flowcontrolv1alpha1.PriorityLevelConfiguration
}

func (d SystemPresetData) EnsureSystemPresetConfiguration() genericapiserver.PostStartHookFunc {
	return func(hookContext genericapiserver.PostStartHookContext) error {
		flowcontrolClientSet := flowcontrolclient.NewForConfigOrDie(hookContext.LoopbackClientConfig)
		// Adding system priority classes is important. If they fail to add, many critical system
		// components may fail and cluster may break.
		err := wait.Poll(1*time.Second, 30*time.Second, func() (done bool, err error) {
			if err != nil {
				utilruntime.HandleError(fmt.Errorf("unable to initialize client: %v", err))
				return false, nil
			}

			for _, flowSchema := range d.FlowSchemas {
				_, err := flowcontrolClientSet.FlowSchemas().Get(flowSchema.Name, metav1.GetOptions{})
				if err != nil {
					if apierrors.IsNotFound(err) {
						_, err := flowcontrolClientSet.FlowSchemas().Create(flowSchema)
						if err != nil && !apierrors.IsAlreadyExists(err) {
							return false, err
						} else if err == nil {
							klog.V(6).Infof("created system preset FlowSchema %s", flowSchema.Name)
						} else {
							klog.V(6).Infof("system preset FlowSchema %s already exists, skipping creating", flowSchema.Name)
						}
					} else {
						// Unable to get the priority class for reasons other than "not found".
						klog.Warningf("unable to get FlowSchema %v: %v. Retrying...", flowSchema.Name, err)
						return false, nil
					}
				}
			}
			for _, priorityLevelConfiguration := range d.PriorityLevelConfigurations {
				_, err := flowcontrolClientSet.PriorityLevelConfigurations().Get(priorityLevelConfiguration.Name, metav1.GetOptions{})
				if err != nil {
					if apierrors.IsNotFound(err) {
						_, err := flowcontrolClientSet.PriorityLevelConfigurations().Create(priorityLevelConfiguration)
						if err != nil && !apierrors.IsAlreadyExists(err) {
							return false, err
						} else if err == nil {
							klog.V(6).Infof("created system preset PriorityLevelConfiguration %s", priorityLevelConfiguration.Name)
						} else {
							klog.V(6).Infof("system preset PriorityLevelConfiguration %s already exists, skipping creating", priorityLevelConfiguration.Name)
						}
					} else if err == nil {
						// Unable to get the priority class for reasons other than "not found".
						klog.Warningf("unable to get PriorityLevelConfiguration %v: %v. Retrying...", priorityLevelConfiguration.Name, err)
						return false, nil
					}
				}
			}
			klog.V(4).Infof("all system flow-control settings are created successfully or already exist.")
			return true, nil
		})
		// if we're never able to make it through initialization, kill the API server.
		if err != nil {
			return fmt.Errorf("unable to add default system flow-control settings: %v", err)
		}
		return nil
	}
}

func (p RESTStorageProvider) GroupName() string {
	return flowcontrol.GroupName
}
