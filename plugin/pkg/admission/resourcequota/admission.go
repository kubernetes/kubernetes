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
	"fmt"
	"io"
	"time"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/quota/install"
	resourcequotaapi "k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota"
	"k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota/validation"
)

func init() {
	admission.RegisterPlugin("ResourceQuota",
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
			// NOTE: we do not provide informers to the registry because admission level decisions
			// does not require us to open watches for all items tracked by quota.
			registry := install.NewRegistry(nil, nil)
			return NewResourceQuota(registry, configuration, 5, make(chan struct{}))
		})
}

// quotaAdmission implements an admission controller that can enforce quota constraints
type quotaAdmission struct {
	*admission.Handler
	config        *resourcequotaapi.Configuration
	stopCh        <-chan struct{}
	registry      quota.Registry
	numEvaluators int
	evaluator     Evaluator
}

var _ = kubeapiserveradmission.WantsInternalClientSet(&quotaAdmission{})

type liveLookupEntry struct {
	expiry time.Time
	items  []*api.ResourceQuota
}

// NewResourceQuota configures an admission controller that can enforce quota constraints
// using the provided registry.  The registry must have the capability to handle group/kinds that
// are persisted by the server this admission controller is intercepting
func NewResourceQuota(registry quota.Registry, config *resourcequotaapi.Configuration, numEvaluators int, stopCh <-chan struct{}) (admission.Interface, error) {
	return &quotaAdmission{
		Handler:       admission.NewHandler(admission.Create, admission.Update),
		stopCh:        stopCh,
		registry:      registry,
		numEvaluators: numEvaluators,
		config:        config,
	}, nil
}

func (a *quotaAdmission) SetInternalClientSet(client internalclientset.Interface) {
	var err error
	quotaAccessor, err := newQuotaAccessor(client)
	if err != nil {
		// TODO handle errors more cleanly
		panic(err)
	}
	go quotaAccessor.Run(a.stopCh)

	a.evaluator = NewQuotaEvaluator(quotaAccessor, a.registry, nil, a.config, a.numEvaluators, a.stopCh)
}

// Validate ensures an authorizer is set.
func (a *quotaAdmission) Validate() error {
	if a.evaluator == nil {
		return fmt.Errorf("missing evaluator")
	}
	return nil
}

// Admit makes admission decisions while enforcing quota
func (a *quotaAdmission) Admit(attr admission.Attributes) (err error) {
	// ignore all operations that correspond to sub-resource actions
	if attr.GetSubresource() != "" {
		return nil
	}
	return a.evaluator.Evaluate(attr)
}
