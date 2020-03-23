/*
Copyright 2014 The Kubernetes Authors.

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

package resourcequota

import (
	"context"
	"fmt"
	"io"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apiserver/pkg/admission"
	genericadmissioninitializer "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	quota "k8s.io/kubernetes/pkg/quota/v1"
	"k8s.io/kubernetes/pkg/quota/v1/generic"
	resourcequotaapi "k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota"
	"k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota/validation"
)

// PluginName is a string with the name of the plugin
const PluginName = "ResourceQuota"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName,
		func(config io.Reader) (admission.Interface, error) {
			// load the configuration provided (if any)
			configuration, err := LoadConfiguration(config)
			if err != nil {
				return nil, err
			}
			// validate the configuration (if any)
			if configuration != nil {
				if errs := validation.ValidateConfiguration(configuration); len(errs) != 0 {
					return nil, errs.ToAggregate()
				}
			}
			return NewResourceQuota(configuration, 5, make(chan struct{}))
		})
}

// QuotaAdmission implements an admission controller that can enforce quota constraints
type QuotaAdmission struct {
	*admission.Handler
	config             *resourcequotaapi.Configuration
	stopCh             <-chan struct{}
	quotaConfiguration quota.Configuration
	numEvaluators      int
	quotaAccessor      *quotaAccessor
	evaluator          Evaluator
}

var _ admission.ValidationInterface = &QuotaAdmission{}
var _ = genericadmissioninitializer.WantsExternalKubeInformerFactory(&QuotaAdmission{})
var _ = genericadmissioninitializer.WantsExternalKubeClientSet(&QuotaAdmission{})
var _ = kubeapiserveradmission.WantsQuotaConfiguration(&QuotaAdmission{})

type liveLookupEntry struct {
	expiry time.Time
	items  []*corev1.ResourceQuota
}

// NewResourceQuota configures an admission controller that can enforce quota constraints
// using the provided registry.  The registry must have the capability to handle group/kinds that
// are persisted by the server this admission controller is intercepting
func NewResourceQuota(config *resourcequotaapi.Configuration, numEvaluators int, stopCh <-chan struct{}) (*QuotaAdmission, error) {
	quotaAccessor, err := newQuotaAccessor()
	if err != nil {
		return nil, err
	}

	return &QuotaAdmission{
		Handler:       admission.NewHandler(admission.Create, admission.Update),
		stopCh:        stopCh,
		numEvaluators: numEvaluators,
		config:        config,
		quotaAccessor: quotaAccessor,
	}, nil
}

// SetExternalKubeClientSet registers the client into QuotaAdmission
func (a *QuotaAdmission) SetExternalKubeClientSet(client kubernetes.Interface) {
	a.quotaAccessor.client = client
}

// SetExternalKubeInformerFactory registers an informer factory into QuotaAdmission
func (a *QuotaAdmission) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	a.quotaAccessor.lister = f.Core().V1().ResourceQuotas().Lister()
}

// SetQuotaConfiguration assigns and initializes configuration and evaluator for QuotaAdmission
func (a *QuotaAdmission) SetQuotaConfiguration(c quota.Configuration) {
	a.quotaConfiguration = c
	a.evaluator = NewQuotaEvaluator(a.quotaAccessor, a.quotaConfiguration.IgnoredResources(), generic.NewRegistry(a.quotaConfiguration.Evaluators()), nil, a.config, a.numEvaluators, a.stopCh)
}

// ValidateInitialization ensures an authorizer is set.
func (a *QuotaAdmission) ValidateInitialization() error {
	if a.quotaAccessor == nil {
		return fmt.Errorf("missing quotaAccessor")
	}
	if a.quotaAccessor.client == nil {
		return fmt.Errorf("missing quotaAccessor.client")
	}
	if a.quotaAccessor.lister == nil {
		return fmt.Errorf("missing quotaAccessor.lister")
	}
	if a.quotaConfiguration == nil {
		return fmt.Errorf("missing quotaConfiguration")
	}
	if a.evaluator == nil {
		return fmt.Errorf("missing evaluator")
	}
	return nil
}

// Validate makes admission decisions while enforcing quota
func (a *QuotaAdmission) Validate(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) (err error) {
	// ignore all operations that correspond to sub-resource actions
	if attr.GetSubresource() != "" {
		return nil
	}
	// ignore all operations that are not namespaced
	if attr.GetNamespace() == "" {
		return nil
	}
	return a.evaluator.Evaluate(attr)
}
