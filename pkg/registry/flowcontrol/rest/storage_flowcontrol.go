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
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	flowcontrolbootstrap "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	flowcontrolclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1beta2"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/flowcontrol"
	flowcontrolapisv1alpha1 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1alpha1"
	flowcontrolapisv1beta1 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1beta1"
	flowcontrolapisv1beta2 "k8s.io/kubernetes/pkg/apis/flowcontrol/v1beta2"
	"k8s.io/kubernetes/pkg/registry/flowcontrol/ensurer"
	flowschemastore "k8s.io/kubernetes/pkg/registry/flowcontrol/flowschema/storage"
	prioritylevelconfigurationstore "k8s.io/kubernetes/pkg/registry/flowcontrol/prioritylevelconfiguration/storage"
)

var _ genericapiserver.PostStartHookProvider = RESTStorageProvider{}

// RESTStorageProvider is a provider of REST storage
type RESTStorageProvider struct{}

// PostStartHookName is the name of the post-start-hook provided by flow-control storage
const PostStartHookName = "priority-and-fairness-config-producer"

// NewRESTStorage creates a new rest storage for flow-control api models.
func (p RESTStorageProvider) NewRESTStorage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (genericapiserver.APIGroupInfo, bool, error) {
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(flowcontrol.GroupName, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)

	if apiResourceConfigSource.VersionEnabled(flowcontrolapisv1alpha1.SchemeGroupVersion) {
		flowControlStorage, err := p.storage(apiResourceConfigSource, restOptionsGetter)
		if err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		}
		apiGroupInfo.VersionedResourcesStorageMap[flowcontrolapisv1alpha1.SchemeGroupVersion.Version] = flowControlStorage
	}

	if apiResourceConfigSource.VersionEnabled(flowcontrolapisv1beta1.SchemeGroupVersion) {
		flowControlStorage, err := p.storage(apiResourceConfigSource, restOptionsGetter)
		if err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		}
		apiGroupInfo.VersionedResourcesStorageMap[flowcontrolapisv1beta1.SchemeGroupVersion.Version] = flowControlStorage
	}

	if apiResourceConfigSource.VersionEnabled(flowcontrolapisv1beta2.SchemeGroupVersion) {
		flowControlStorage, err := p.storage(apiResourceConfigSource, restOptionsGetter)
		if err != nil {
			return genericapiserver.APIGroupInfo{}, false, err
		}
		apiGroupInfo.VersionedResourcesStorageMap[flowcontrolapisv1beta2.SchemeGroupVersion.Version] = flowControlStorage
	}

	return apiGroupInfo, true, nil
}

func (p RESTStorageProvider) storage(apiResourceConfigSource serverstorage.APIResourceConfigSource, restOptionsGetter generic.RESTOptionsGetter) (map[string]rest.Storage, error) {
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

// PostStartHook returns the hook func that launches the config provider
func (p RESTStorageProvider) PostStartHook() (string, genericapiserver.PostStartHookFunc, error) {
	return PostStartHookName, ensureAPFBootstrapConfiguration, nil
}

func ensureAPFBootstrapConfiguration(hookContext genericapiserver.PostStartHookContext) error {
	clientset, err := flowcontrolclient.NewForConfig(hookContext.LoopbackClientConfig)
	if err != nil {
		return fmt.Errorf("failed to initialize clientset for APF - %w", err)
	}

	// get a derived context that gets cancelled after 5m or
	// when the StopCh gets closed, whichever happens first.
	ctx, cancel := contextFromChannelAndMaxWaitDuration(hookContext.StopCh, 5*time.Minute)
	defer cancel()

	err = wait.PollImmediateUntilWithContext(
		ctx,
		time.Second,
		func(context.Context) (bool, error) {
			if err := ensure(clientset); err != nil {
				klog.ErrorS(err, "APF bootstrap ensurer ran into error, will retry later")
				return false, nil
			}
			return true, nil
		})
	if err != nil {
		return fmt.Errorf("unable to initialize APF bootstrap configuration")
	}

	// we have successfully initialized the bootstrap configuration, now we
	// spin up a goroutine which reconciles the bootstrap configuration periodically.
	go func() {
		wait.PollImmediateUntil(
			time.Minute,
			func() (bool, error) {
				if err := ensure(clientset); err != nil {
					klog.ErrorS(err, "APF bootstrap ensurer ran into error, will retry later")
				}
				// always auto update both suggested and mandatory configuration
				return false, nil
			}, hookContext.StopCh)
		klog.Info("APF bootstrap ensurer is exiting")
	}()

	return nil
}

func ensure(clientset flowcontrolclient.FlowcontrolV1beta2Interface) error {
	if err := ensureSuggestedConfiguration(clientset); err != nil {
		// We should not attempt creation of mandatory objects if ensuring the suggested
		// configuration resulted in an error.
		// This only happens when the stop channel is closed.
		return fmt.Errorf("failed ensuring suggested settings - %w", err)
	}

	if err := ensureMandatoryConfiguration(clientset); err != nil {
		return fmt.Errorf("failed ensuring mandatory settings - %w", err)
	}

	if err := removeConfiguration(clientset); err != nil {
		return fmt.Errorf("failed to delete removed settings - %w", err)
	}

	return nil
}

func ensureSuggestedConfiguration(clientset flowcontrolclient.FlowcontrolV1beta2Interface) error {
	fsEnsurer := ensurer.NewSuggestedFlowSchemaEnsurer(clientset.FlowSchemas())
	if err := fsEnsurer.Ensure(flowcontrolbootstrap.SuggestedFlowSchemas); err != nil {
		return err
	}

	plEnsurer := ensurer.NewSuggestedPriorityLevelEnsurerEnsurer(clientset.PriorityLevelConfigurations())
	return plEnsurer.Ensure(flowcontrolbootstrap.SuggestedPriorityLevelConfigurations)
}

func ensureMandatoryConfiguration(clientset flowcontrolclient.FlowcontrolV1beta2Interface) error {
	fsEnsurer := ensurer.NewMandatoryFlowSchemaEnsurer(clientset.FlowSchemas())
	if err := fsEnsurer.Ensure(flowcontrolbootstrap.MandatoryFlowSchemas); err != nil {
		return err
	}

	plEnsurer := ensurer.NewMandatoryPriorityLevelEnsurer(clientset.PriorityLevelConfigurations())
	return plEnsurer.Ensure(flowcontrolbootstrap.MandatoryPriorityLevelConfigurations)
}

func removeConfiguration(clientset flowcontrolclient.FlowcontrolV1beta2Interface) error {
	if err := removeFlowSchema(clientset.FlowSchemas()); err != nil {
		return err
	}

	return removePriorityLevel(clientset.PriorityLevelConfigurations())
}

func removeFlowSchema(client flowcontrolclient.FlowSchemaInterface) error {
	bootstrap := append(flowcontrolbootstrap.MandatoryFlowSchemas, flowcontrolbootstrap.SuggestedFlowSchemas...)
	candidates, err := ensurer.GetFlowSchemaRemoveCandidate(client, bootstrap)
	if err != nil {
		return err
	}
	if len(candidates) == 0 {
		return nil
	}

	fsRemover := ensurer.NewFlowSchemaRemover(client)
	return fsRemover.Remove(candidates)
}

func removePriorityLevel(client flowcontrolclient.PriorityLevelConfigurationInterface) error {
	bootstrap := append(flowcontrolbootstrap.MandatoryPriorityLevelConfigurations, flowcontrolbootstrap.SuggestedPriorityLevelConfigurations...)
	candidates, err := ensurer.GetPriorityLevelRemoveCandidate(client, bootstrap)
	if err != nil {
		return err
	}
	if len(candidates) == 0 {
		return nil
	}

	plRemover := ensurer.NewPriorityLevelRemover(client)
	return plRemover.Remove(candidates)
}

// contextFromChannelAndMaxWaitDuration returns a Context that is bound to the
// specified channel and the wait duration. The derived context will be
// cancelled when the specified channel stopCh is closed or the maximum wait
// duration specified in maxWait elapses, whichever happens first.
//
// Note the caller must *always* call the CancelFunc, otherwise resources may be leaked.
func contextFromChannelAndMaxWaitDuration(stopCh <-chan struct{}, maxWait time.Duration) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		defer cancel()

		select {
		case <-stopCh:
		case <-time.After(maxWait):

		// the caller can explicitly cancel the context which is an
		// indication to us to exit the goroutine immediately.
		// Note that we are calling cancel more than once when we are here,
		// CancelFunc is idempotent and we expect no ripple effects here.
		case <-ctx.Done():
		}
	}()
	return ctx, cancel
}
