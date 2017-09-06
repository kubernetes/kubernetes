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

package eventratelimit

import (
	"io"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/api"
	eventratelimitapi "k8s.io/kubernetes/plugin/pkg/admission/eventratelimit/apis/eventratelimit"
	"k8s.io/kubernetes/plugin/pkg/admission/eventratelimit/apis/eventratelimit/validation"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register("EventRateLimit",
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
			return newEventRateLimit(configuration, realClock{})
		})
}

// eventRateLimitAdmission implements an admission controller that can enforce event rate limits
type eventRateLimitAdmission struct {
	*admission.Handler
	// limitEnforcers is the collection of limit enforcers. There is one limit enforcer for each
	// active limit type. As there are 4 limit types, the length of the array will be at most 4.
	// The array is read-only after construction.
	limitEnforcers []*limitEnforcer
}

// newEventRateLimit configures an admission controller that can enforce event rate limits
func newEventRateLimit(config *eventratelimitapi.Configuration, clock flowcontrol.Clock) (admission.Interface, error) {
	limitEnforcers := make([]*limitEnforcer, 0, len(config.Limits))
	for _, limitConfig := range config.Limits {
		enforcer, err := newLimitEnforcer(limitConfig, clock)
		if err != nil {
			return nil, err
		}
		limitEnforcers = append(limitEnforcers, enforcer)
	}

	eventRateLimitAdmission := &eventRateLimitAdmission{
		Handler:        admission.NewHandler(admission.Create, admission.Update),
		limitEnforcers: limitEnforcers,
	}

	return eventRateLimitAdmission, nil
}

// Admit makes admission decisions while enforcing event rate limits
func (a *eventRateLimitAdmission) Admit(attr admission.Attributes) (err error) {
	// ignore all operations that do not correspond to an Event kind
	if attr.GetKind().GroupKind() != api.Kind("Event") {
		return nil
	}

	var rejectionError error
	// give each limit enforcer a chance to reject the event
	for _, enforcer := range a.limitEnforcers {
		if err := enforcer.accept(attr); err != nil {
			rejectionError = err
		}
	}

	return rejectionError
}
