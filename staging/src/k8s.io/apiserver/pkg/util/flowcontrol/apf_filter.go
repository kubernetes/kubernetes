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

package flowcontrol

import (
	"context"
	"strconv"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/apiserver/pkg/util/flowcontrol/counter"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	fqs "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/queueset"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/klog/v2"

	flowcontrol "k8s.io/api/flowcontrol/v1beta1"
	flowcontrolclient "k8s.io/client-go/kubernetes/typed/flowcontrol/v1beta1"
)

// ConfigConsumerAsFieldManager is how the config consuminng
// controller appears in an ObjectMeta ManagedFieldsEntry.Manager
const ConfigConsumerAsFieldManager = "api-priority-and-fairness-config-consumer-v1"

// Interface defines how the API Priority and Fairness filter interacts with the underlying system.
type Interface interface {
	// Handle takes care of queuing and dispatching a request
	// characterized by the given digest.  The given `noteFn` will be
	// invoked with the results of request classification.  If the
	// request is queued then `queueNoteFn` will be called twice,
	// first with `true` and then with `false`; otherwise
	// `queueNoteFn` will not be called at all.  If Handle decides
	// that the request should be executed then `execute()` will be
	// invoked once to execute the request; otherwise `execute()` will
	// not be invoked.
	Handle(ctx context.Context,
		requestDigest RequestDigest,
		noteFn func(fs *flowcontrol.FlowSchema, pl *flowcontrol.PriorityLevelConfiguration),
		queueNoteFn fq.QueueNoteFn,
		execFn func(),
	)

	// MaintainObservations is a helper for maintaining statistics.
	MaintainObservations(stopCh <-chan struct{})

	// Run monitors config objects from the main apiservers and causes
	// any needed changes to local behavior.  This method ceases
	// activity and returns after the given channel is closed.
	Run(stopCh <-chan struct{}) error

	// Install installs debugging endpoints to the web-server.
	Install(c *mux.PathRecorderMux)
}

// This request filter implements https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/1040-priority-and-fairness/README.md

// New creates a new instance to implement API priority and fairness
func New(
	informerFactory kubeinformers.SharedInformerFactory,
	flowcontrolClient flowcontrolclient.FlowcontrolV1beta1Interface,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
) Interface {
	grc := counter.NoOp{}
	clk := clock.RealClock{}
	return NewTestable(TestableConfig{
		Name:                   "Controller",
		Clock:                  clk,
		AsFieldManager:         ConfigConsumerAsFieldManager,
		FoundToDangling:        func(found bool) bool { return !found },
		InformerFactory:        informerFactory,
		FlowcontrolClient:      flowcontrolClient,
		ServerConcurrencyLimit: serverConcurrencyLimit,
		RequestWaitLimit:       requestWaitLimit,
		ObsPairGenerator:       metrics.PriorityLevelConcurrencyObserverPairGenerator,
		QueueSetFactory:        fqs.NewQueueSetFactory(clk, grc),
	})
}

// TestableConfig carries the parameters to an implementation that is testable
type TestableConfig struct {
	// Name of the controller
	Name string

	// Clock to use in timing deliberate delays
	Clock clock.PassiveClock

	// AsFieldManager is the string to use in the metadata for
	// server-side apply.  Normally this is
	// `ConfigConsumerAsFieldManager`.  This is exposed as a parameter
	// so that a test of competing controllers can supply different
	// values.
	AsFieldManager string

	// FoundToDangling maps the boolean indicating whether a
	// FlowSchema's referenced PLC exists to the boolean indicating
	// that FlowSchema's status should indicate a dangling reference.
	// This is a parameter so that we can write tests of what happens
	// when servers disagree on that bit of Status.
	FoundToDangling func(bool) bool

	// InformerFactory to use in building the controller
	InformerFactory kubeinformers.SharedInformerFactory

	// FlowcontrolClient to use for manipulating config objects
	FlowcontrolClient flowcontrolclient.FlowcontrolV1beta1Interface

	// ServerConcurrencyLimit for the controller to enforce
	ServerConcurrencyLimit int

	// RequestWaitLimit configured on the server
	RequestWaitLimit time.Duration

	// ObsPairGenerator for metrics
	ObsPairGenerator metrics.TimedObserverPairGenerator

	// QueueSetFactory for the queuing implementation
	QueueSetFactory fq.QueueSetFactory
}

// NewTestable is extra flexible to facilitate testing
func NewTestable(config TestableConfig) Interface {
	return newTestableController(config)
}

func (cfgCtlr *configController) Handle(ctx context.Context, requestDigest RequestDigest,
	noteFn func(fs *flowcontrol.FlowSchema, pl *flowcontrol.PriorityLevelConfiguration),
	queueNoteFn fq.QueueNoteFn,
	execFn func()) {
	fs, pl, isExempt, req, startWaitingTime := cfgCtlr.startRequest(ctx, requestDigest, queueNoteFn)
	queued := startWaitingTime != time.Time{}
	noteFn(fs, pl)
	if req == nil {
		if queued {
			metrics.ObserveWaitingDuration(ctx, pl.Name, fs.Name, strconv.FormatBool(req != nil), time.Since(startWaitingTime))
		}
		klog.V(7).Infof("Handle(%#+v) => fsName=%q, distMethod=%#+v, plName=%q, isExempt=%v, reject", requestDigest, fs.Name, fs.Spec.DistinguisherMethod, pl.Name, isExempt)
		return
	}
	klog.V(7).Infof("Handle(%#+v) => fsName=%q, distMethod=%#+v, plName=%q, isExempt=%v, queued=%v", requestDigest, fs.Name, fs.Spec.DistinguisherMethod, pl.Name, isExempt, queued)
	var executed bool
	idle, panicking := true, true
	defer func() {
		klog.V(7).Infof("Handle(%#+v) => fsName=%q, distMethod=%#+v, plName=%q, isExempt=%v, queued=%v, Finish() => panicking=%v idle=%v",
			requestDigest, fs.Name, fs.Spec.DistinguisherMethod, pl.Name, isExempt, queued, panicking, idle)
		if idle {
			cfgCtlr.maybeReap(pl.Name)
		}
	}()
	idle = req.Finish(func() {
		if queued {
			metrics.ObserveWaitingDuration(ctx, pl.Name, fs.Name, strconv.FormatBool(req != nil), time.Since(startWaitingTime))
		}
		metrics.AddDispatch(ctx, pl.Name, fs.Name)
		executed = true
		startExecutionTime := time.Now()
		defer func() {
			metrics.ObserveExecutionDuration(ctx, pl.Name, fs.Name, time.Since(startExecutionTime))
		}()
		execFn()
	})
	if queued && !executed {
		metrics.ObserveWaitingDuration(ctx, pl.Name, fs.Name, strconv.FormatBool(req != nil), time.Since(startWaitingTime))
	}
	panicking = false
}
