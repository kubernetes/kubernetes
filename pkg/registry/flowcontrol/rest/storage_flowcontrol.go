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
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	flowcontrolbootstrap "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	flowcontrolclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1alpha1"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
	flowcontrolapisv1alpha1 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1alpha1"
	flowschemastore "k8s.io/kubernetes/pkg/registry/flowcontrol/flowschema/storage"
	prioritylevelconfigurationstore "k8s.io/kubernetes/pkg/registry/flowcontrol/prioritylevelconfiguration/storage"
)

var _ genericapiserver.PostStartHookProvider = RESTStorageProvider{}

// RESTStorageProvider is a provider of REST storage
type RESTStorageProvider struct{}

// PostStartHookName is the name of the post-start-hook provided by flow-control storage
const PostStartHookName = "apiserver/bootstrap-system-flowcontrol-configuration"

// NewRESTStorage creates a new rest storage for flow-control api models.
func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(flowcontrol.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)

	if apiResourceConfigSource.VersionEnabled(flowcontrolv1alpha1.SchemeGroupVersion) {
		flowControlStorage, err := p.v1alpha1Storage(apiResourceConfigSource, restOptionsGetter)
		if err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		}
		apiGroupInfo.VersionedResourcesStorageMap[flowcontrolv1alpha1.SchemeGroupVersion.Version] = flowControlStorage
	}
	return apiGroupInfo, true, nil
}

func (p RESTStorageProvider) v1alpha1Storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
	storage := map[string]rest.Storage{}

	// flow-schema
	flowSchemaStorage, flowSchemaStatusStorage, err := flowschemastore.NewREST(restOptionsGetter)
	if err != nil {
		return nil, err
	}
	storage["flowschemas"] = flowSchemaStorage
	storage["flowschemas/status"] = flowSchemaStatusStorage

	// priority-level-configuration
	priorityLevelConfigurationStorage, priorityLevelConfigurationStatusStorage, err := prioritylevelconfigurationstore.NewREST(restOptionsGetter)
	if err != nil {
		return nil, err
	}
	storage["prioritylevelconfigurations"] = priorityLevelConfigurationStorage
	storage["prioritylevelconfigurations/status"] = priorityLevelConfigurationStatusStorage

	return storage, nil
}

// GroupName returns group name of the storage
func (p RESTStorageProvider) GroupName() string {
	return flowcontrol.GroupName
}

func (p RESTStorageProvider) PostStartHook() (string, genericapiserver.PostStartHookFunc, error) {
	return PostStartHookName, func(hookContext genericapiserver.PostStartHookContext) error {
		flowcontrolClientSet := flowcontrolclient.NewForConfigOrDie(hookContext.LoopbackClientConfig)
		go func() {
			const retryCreatingSuggestedSettingsInterval = time.Second
			_ = wait.PollImmediateUntil(
				retryCreatingSuggestedSettingsInterval,
				func() (bool, error) {
					shouldEnsureSuggested, err := lastMandatoryExists(flowcontrolClientSet)
					if err != nil {
						klog.Errorf("failed getting exempt flow-schema, will retry later: %v", err)
						return false, nil
					}
					if !shouldEnsureSuggested {
						return true, nil
					}
					err = ensure(
						flowcontrolClientSet,
						flowcontrolbootstrap.SuggestedFlowSchemas,
						flowcontrolbootstrap.SuggestedPriorityLevelConfigurations)
					if err != nil {
						klog.Errorf("failed ensuring suggested settings, will retry later: %v", err)
						return false, nil
					}
					return true, nil
				},
				hookContext.StopCh)
			const retryCreatingMandatorySettingsInterval = time.Minute
			_ = wait.PollImmediateUntil(
				retryCreatingMandatorySettingsInterval,
				func() (bool, error) {
					if err := upgrade(
						flowcontrolClientSet,
						flowcontrolbootstrap.MandatoryFlowSchemas,
						// Note: the "exempt" priority-level is supposed tobe the last item in the pre-defined
						// list, so that a crash in the midst of the first kube-apiserver startup does not prevent
						// the full initial set of objects from being created.
						flowcontrolbootstrap.MandatoryPriorityLevelConfigurations,
					); err != nil {
						klog.Errorf("failed creating mandatory flowcontrol settings: %v", err)
						return false, nil
					}
					return false, nil // always retry
				},
				hookContext.StopCh)
		}()
		return nil
	}, nil

}

// Returns false if there's a "exempt" priority-level existing in the cluster, otherwise returns a true
// if the "exempt" priority-level is not found.
func lastMandatoryExists(flowcontrolClientSet flowcontrolclient.FlowcontrolV1alpha1Interface) (bool, error) {
	if _, err := flowcontrolClientSet.PriorityLevelConfigurations().Get(flowcontrol.PriorityLevelConfigurationNameExempt, metav1.GetOptions{}); err != nil {
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	}
	return false, nil
}

func ensure(flowcontrolClientSet flowcontrolclient.FlowcontrolV1alpha1Interface, flowSchemas []*flowcontrolv1alpha1.FlowSchema, priorityLevels []*flowcontrolv1alpha1.PriorityLevelConfiguration) error {
	for _, flowSchema := range flowSchemas {
		_, err := flowcontrolClientSet.FlowSchemas().Create(flowSchema)
		if apierrors.IsAlreadyExists(err) {
			klog.V(3).Infof("system preset FlowSchema %s already exists, skipping creating", flowSchema.Name)
			continue
		}
		if err != nil {
			return fmt.Errorf("cannot create FlowSchema %s due to %v", flowSchema.Name, err)
		}
		klog.V(3).Infof("created system preset FlowSchema %s", flowSchema.Name)
	}
	for _, priorityLevelConfiguration := range priorityLevels {
		_, err := flowcontrolClientSet.PriorityLevelConfigurations().Create(priorityLevelConfiguration)
		if apierrors.IsAlreadyExists(err) {
			klog.V(3).Infof("system preset PriorityLevelConfiguration %s already exists, skipping creating", priorityLevelConfiguration.Name)
			continue
		}
		if err != nil {
			return fmt.Errorf("cannot create PriorityLevelConfiguration %s due to %v", priorityLevelConfiguration.Name, err)
		}
		klog.V(3).Infof("created system preset PriorityLevelConfiguration %s", priorityLevelConfiguration.Name)
	}
	return nil
}

func upgrade(flowcontrolClientSet flowcontrolclient.FlowcontrolV1alpha1Interface, flowSchemas []*flowcontrolv1alpha1.FlowSchema, priorityLevels []*flowcontrolv1alpha1.PriorityLevelConfiguration) error {
	for _, expectedFlowSchema := range flowSchemas {
		actualFlowSchema, err := flowcontrolClientSet.FlowSchemas().Get(expectedFlowSchema.Name, metav1.GetOptions{})
		if err == nil {
			// TODO(yue9944882): extract existing version from label and compare
			// TODO(yue9944882): create w/ version string attached
			identical, err := flowSchemaHasWrongSpec(expectedFlowSchema, actualFlowSchema)
			if err != nil {
				return fmt.Errorf("failed checking if mandatory FlowSchema %s is up-to-date due to %v, will retry later", expectedFlowSchema.Name, err)
			}
			if !identical {
				if _, err := flowcontrolClientSet.FlowSchemas().Update(expectedFlowSchema); err != nil {
					return fmt.Errorf("failed upgrading mandatory FlowSchema %s due to %v, will retry later", expectedFlowSchema.Name, err)
				}
			}
			continue
		}
		if !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed getting FlowSchema %s due to %v, will retry later", expectedFlowSchema.Name, err)
		}
		_, err = flowcontrolClientSet.FlowSchemas().Create(expectedFlowSchema)
		if apierrors.IsAlreadyExists(err) {
			klog.V(3).Infof("system preset FlowSchema %s already exists, skipping creating", expectedFlowSchema.Name)
			continue
		}
		if err != nil {
			return fmt.Errorf("cannot create FlowSchema %s due to %v", expectedFlowSchema.Name, err)
		}
		klog.V(3).Infof("created system preset FlowSchema %s", expectedFlowSchema.Name)
	}
	for _, expectedPriorityLevelConfiguration := range priorityLevels {
		actualPriorityLevelConfiguration, err := flowcontrolClientSet.PriorityLevelConfigurations().Get(expectedPriorityLevelConfiguration.Name, metav1.GetOptions{})
		if err == nil {
			// TODO(yue9944882): extract existing version from label and compare
			// TODO(yue9944882): create w/ version string attached
			identical, err := priorityLevelHasWrongSpec(expectedPriorityLevelConfiguration, actualPriorityLevelConfiguration)
			if err != nil {
				return fmt.Errorf("failed checking if mandatory PriorityLevelConfiguration %s is up-to-date due to %v, will retry later", expectedPriorityLevelConfiguration.Name, err)
			}
			if !identical {
				if _, err := flowcontrolClientSet.PriorityLevelConfigurations().Update(expectedPriorityLevelConfiguration); err != nil {
					return fmt.Errorf("failed upgrading mandatory PriorityLevelConfiguration %s due to %v, will retry later", expectedPriorityLevelConfiguration.Name, err)
				}
			}
			continue
		}
		if !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed getting PriorityLevelConfiguration %s due to %v, will retry later", expectedPriorityLevelConfiguration.Name, err)
		}
		_, err = flowcontrolClientSet.PriorityLevelConfigurations().Create(expectedPriorityLevelConfiguration)
		if apierrors.IsAlreadyExists(err) {
			klog.V(3).Infof("system preset PriorityLevelConfiguration %s already exists, skipping creating", expectedPriorityLevelConfiguration.Name)
			continue
		}
		if err != nil {
			return fmt.Errorf("cannot create PriorityLevelConfiguration %s due to %v", expectedPriorityLevelConfiguration.Name, err)
		}
		klog.V(3).Infof("created system preset PriorityLevelConfiguration %s", expectedPriorityLevelConfiguration.Name)
	}
	return nil
}

func flowSchemaHasWrongSpec(expected, actual *flowcontrolv1alpha1.FlowSchema) (bool, error) {
	copiedExpectedFlowSchema := expected.DeepCopy()
	flowcontrolapisv1alpha1.SetObjectDefaults_FlowSchema(copiedExpectedFlowSchema)
	return !equality.Semantic.DeepEqual(copiedExpectedFlowSchema.Spec, actual.Spec), nil
}

func priorityLevelHasWrongSpec(expected, actual *flowcontrolv1alpha1.PriorityLevelConfiguration) (bool, error) {
	copiedExpectedPriorityLevel := expected.DeepCopy()
	flowcontrolapisv1alpha1.SetObjectDefaults_PriorityLevelConfiguration(copiedExpectedPriorityLevel)
	return !equality.Semantic.DeepEqual(copiedExpectedPriorityLevel.Spec, actual.Spec), nil
}
