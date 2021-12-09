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

package testing

import (
	"context"

	"k8s.io/apiserver/pkg/util/flowcontrol/debug"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	fcrequest "k8s.io/apiserver/pkg/util/flowcontrol/request"
)

// NewNoRestraintFactory makes a QueueSetFactory that produces
// QueueSets that exert no restraint --- every request is dispatched
// for execution as soon as it arrives.
func NewNoRestraintFactory() fq.QueueSetFactory {
	return noRestraintFactory{}
}

type noRestraintFactory struct{}

type noRestraintCompleter struct{}

type noRestraint struct{}

type noRestraintRequest struct{}

func (noRestraintFactory) BeginConstruction(fq.QueuingConfig, metrics.RatioedChangeObserverPair, metrics.RatioedChangeObserver) (fq.QueueSetCompleter, error) {
	return noRestraintCompleter{}, nil
}

func (noRestraintCompleter) Complete(dCfg fq.DispatchingConfig) fq.QueueSet {
	return noRestraint{}
}

func (noRestraint) BeginConfigChange(qCfg fq.QueuingConfig) (fq.QueueSetCompleter, error) {
	return noRestraintCompleter{}, nil
}

func (noRestraint) IsIdle() bool {
	return false
}

func (noRestraint) StartRequest(ctx context.Context, workEstimate *fcrequest.WorkEstimate, hashValue uint64, flowDistinguisher, fsName string, descr1, descr2 interface{}, queueNoteFn fq.QueueNoteFn) (fq.Request, bool) {
	return noRestraintRequest{}, false
}

func (noRestraint) UpdateObservations() {
}

func (noRestraint) Dump(bool) debug.QueueSetDump {
	return debug.QueueSetDump{}
}

func (noRestraintRequest) Finish(execute func()) (idle bool) {
	execute()
	return false
}
