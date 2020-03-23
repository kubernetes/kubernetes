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
	"context"
	"io"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/util/flowcontrol"
	api "k8s.io/kubernetes/pkg/apis/core"
	eventratelimitapi "k8s.io/kubernetes/plugin/pkg/admission/eventratelimit/apis/eventratelimit"
	"k8s.io/kubernetes/plugin/pkg/admission/eventratelimit/apis/eventratelimit/validation"
)

// PluginName indicates name of admission plugin.
const PluginName = "EventRateLimit"

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
			return newEventRateLimit(configuration, realClock{})
		})
}

// Plugin implements an admission controller that can enforce event rate limits
type Plugin struct {
	*admission.Handler
	// limitEnforcers is the collection of limit enforcers. There is one limit enforcer for each
	// active limit type. As there are 4 limit types, the length of the array will be at most 4.
	// The array is read-only after construction.
	limitEnforcers []*limitEnforcer
}

var _ admission.ValidationInterface = &Plugin{}

// newEventRateLimit configures an admission controller that can enforce event rate limits
func newEventRateLimit(config *eventratelimitapi.Configuration, clock flowcontrol.Clock) (*Plugin, error) {
	limitEnforcers := make([]*limitEnforcer, 0, len(config.Limits))
	for _, limitConfig := range config.Limits {
		enforcer, err := newLimitEnforcer(limitConfig, clock)
		if err != nil {
			return nil, err
		}
		limitEnforcers = append(limitEnforcers, enforcer)
	}

	eventRateLimitAdmission := &Plugin{
		Handler:        admission.NewHandler(admission.Create, admission.Update),
		limitEnforcers: limitEnforcers,
	}

	return eventRateLimitAdmission, nil
}

// Validate makes admission decisions while enforcing event rate limits
func (a *Plugin) Validate(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) (err error) {
	// ignore all operations that do not correspond to an Event kind
	if attr.GetKind().GroupKind() != api.Kind("Event") {
		return nil
	}

	// ignore all requests that specify dry-run
	// because they don't correspond to any calls to etcd,
	// they should not be affected by the ratelimit
	if attr.IsDryRun() {
		return nil
	}

	var errors []error
	// give each limit enforcer a chance to reject the event
	for _, enforcer := range a.limitEnforcers {
		if err := enforcer.accept(attr); err != nil {
			errors = append(errors, err)
		}
	}

	if aggregatedErr := utilerrors.NewAggregate(errors); aggregatedErr != nil {
		return apierrors.NewTooManyRequestsError(aggregatedErr.Error())
	}

	return nil
}
