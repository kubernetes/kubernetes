/*
Copyright 2018 The Kubernetes Authors.

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

package dynamic

import (
	"fmt"
	"time"

	auditregv1alpha1 "k8s.io/api/auditregistration/v1alpha1"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/audit/policy"
	auditutil "k8s.io/apiserver/pkg/audit/util"
	"k8s.io/apiserver/pkg/util/webhook"
	bufferedplugin "k8s.io/apiserver/plugin/pkg/audit/buffered"
	enforcedplugin "k8s.io/apiserver/plugin/pkg/audit/dynamic/enforced"
	webhookplugin "k8s.io/apiserver/plugin/pkg/audit/webhook"
)

// TODO: find a common place for all the default retry backoffs
const retryBackoff = 500 * time.Millisecond

// factory builds a delegate from an AuditSink
type factory struct {
	config               *Config
	webhookClientManager webhook.ClientManager
	sink                 *auditregv1alpha1.AuditSink
}

// BuildDelegate creates a delegate from the AuditSink object
func (f *factory) BuildDelegate() (*delegate, error) {
	backend, err := f.buildWebhookBackend()
	if err != nil {
		return nil, err
	}
	backend = f.applyEnforcedOpts(backend)
	backend = f.applyBufferedOpts(backend)
	ch := make(chan struct{})
	return &delegate{
		Backend:       backend,
		configuration: f.sink,
		stopChan:      ch,
	}, nil
}

func (f *factory) buildWebhookBackend() (audit.Backend, error) {
	hookClient := auditutil.HookClientConfigForSink(f.sink)
	client, err := f.webhookClientManager.HookClient(hookClient)
	if err != nil {
		return nil, fmt.Errorf("could not create webhook client: %v", err)
	}
	backend := webhookplugin.NewDynamicBackend(client, retryBackoff)
	return backend, nil
}

func (f *factory) applyEnforcedOpts(delegate audit.Backend) audit.Backend {
	pol := policy.ConvertDynamicPolicyToInternal(&f.sink.Spec.Policy)
	checker := policy.NewChecker(pol)
	eb := enforcedplugin.NewBackend(delegate, checker)
	return eb
}

func (f *factory) applyBufferedOpts(delegate audit.Backend) audit.Backend {
	bc := f.config.BufferedConfig
	tc := f.sink.Spec.Webhook.Throttle
	if tc != nil {
		bc.ThrottleEnable = true
		if tc.Burst != nil {
			bc.ThrottleBurst = int(*tc.Burst)
		}
		if tc.QPS != nil {
			bc.ThrottleQPS = float32(*tc.QPS)
		}
	} else {
		bc.ThrottleEnable = false
	}
	return bufferedplugin.NewBackend(delegate, *bc)
}
