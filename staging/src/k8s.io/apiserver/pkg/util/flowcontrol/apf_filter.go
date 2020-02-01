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
	"crypto/sha256"
	"encoding/binary"
	"hash/crc64"
	"strconv"
	"time"

	// TODO: decide whether to use the existing metrics, which
	// categorize according to mutating vs readonly, or make new
	// metrics because this filter does not pay attention to that
	// distinction

	// "k8s.io/apiserver/pkg/endpoints/metrics"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apiserver/pkg/util/flowcontrol/counter"
	fq "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing"
	fqs "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/queueset"
	fcfc "k8s.io/apiserver/pkg/util/flowcontrol/filterconfig"
	"k8s.io/apiserver/pkg/util/flowcontrol/metrics"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/klog"

	fctypesv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	fcclientv1a1 "k8s.io/client-go/kubernetes/typed/flowcontrol/v1alpha1"
)

// Interface defines how the API Priority and Fairness filter interacts with the underlying system.
type Interface interface {
	// Wait decides what to do about the request with the given digest
	// and, if appropriate, enqueues that request and waits for it to be
	// dequeued before returning.  If `execute == false` then the request
	// is being rejected.  If `execute == true` then the caller should
	// handle the request and then call `afterExecute()`.
	Wait(ctx context.Context, requestDigest fcfc.RequestDigest) (execute bool, afterExecute func())

	// Run monitors config objects from the main apiservers and causes
	// any needed changes to local behavior.  This method ceases
	// activity and returns after the given channel is closed.
	Run(stopCh <-chan struct{}) error
}

// This request filter implements https://github.com/kubernetes/enhancements/blob/master/keps/sig-api-machinery/20190228-priority-and-fairness.md

type implementation struct {
	ctl fcfc.Controller
}

// New creates a new instance to implement API priority and fairness
func New(
	informerFactory kubeinformers.SharedInformerFactory,
	flowcontrolClient fcclientv1a1.FlowcontrolV1alpha1Interface,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
) Interface {
	grc := counter.NoOp{}
	return NewTestable(
		informerFactory,
		flowcontrolClient,
		serverConcurrencyLimit,
		requestWaitLimit,
		fqs.NewQueueSetFactory(&clock.RealClock{}, grc),
	)
}

// NewTestable is extra flexible to facilitate testing
func NewTestable(
	informerFactory kubeinformers.SharedInformerFactory,
	flowcontrolClient fcclientv1a1.FlowcontrolV1alpha1Interface,
	serverConcurrencyLimit int,
	requestWaitLimit time.Duration,
	queueSetFactory fq.QueueSetFactory,
) Interface {
	return &implementation{ctl: fcfc.NewTestableController(informerFactory, flowcontrolClient, serverConcurrencyLimit, requestWaitLimit, queueSetFactory)}
}

func (impl *implementation) Run(stopCh <-chan struct{}) error {
	return impl.ctl.Run(stopCh)
}

func (impl *implementation) Wait(ctx context.Context, requestDigest fcfc.RequestDigest) (bool, func()) {
	startWaitingTime := time.Now()

	// 1. classify the request
	fsName, distinguisherMethod, plName, startFn := impl.ctl.Match(requestDigest)

	// 2. early out for exempt
	if startFn == nil {
		klog.V(7).Infof("Serving requestInfo=%#+v, userInfo=%#+v, fs=%s, pl=%s without delay", requestDigest.RequestInfo, requestDigest.User, fsName, plName)
		startExecutionTime := time.Now()
		return true, func() {
			metrics.ObserveExecutionDuration(plName, fsName, time.Now().Sub(startExecutionTime))
		}
	}

	// 3. computing hash
	flowDistinguisher := computeFlowDistinguisher(requestDigest, distinguisherMethod)
	hashValue := hashFlowID(fsName, flowDistinguisher)

	// 4. queuing
	execute, afterExecute := startFn(ctx, hashValue)

	// 5. execute or reject
	metrics.ObserveWaitingDuration(plName, fsName, strconv.FormatBool(execute), time.Now().Sub(startWaitingTime))
	if !execute {
		klog.V(7).Infof("Rejecting requestInfo=%#+v, userInfo=%#+v, fs=%s, pl=%s after fair queuing", requestDigest.RequestInfo, requestDigest.User, fsName, plName)
		return false, func() {}
	}
	klog.V(7).Infof("Serving requestInfo=%#+v, userInfo=%#+v, fs=%s, pl=%s after fair queuing", requestDigest.RequestInfo, requestDigest.User, fsName, plName)
	startExecutionTime := time.Now()
	return execute, func() {
		metrics.ObserveExecutionDuration(plName, fsName, time.Now().Sub(startExecutionTime))
		afterExecute()
	}
}

// computeFlowDistinguisher extracts the flow distinguisher according to the given method
func computeFlowDistinguisher(rd fcfc.RequestDigest, method *fctypesv1a1.FlowDistinguisherMethod) string {
	if method == nil {
		return ""
	}
	switch method.Type {
	case fctypesv1a1.FlowDistinguisherMethodByUserType:
		return rd.User.GetName()
	case fctypesv1a1.FlowDistinguisherMethodByNamespaceType:
		return rd.RequestInfo.Namespace
	default:
		// this line shall never reach
		panic("invalid flow-distinguisher method")
	}
}

var hashByCRC bool

// hashFlowID hashes the inputs into 64-bits
func hashFlowID(fsName, fDistinguisher string) uint64 {
	if hashByCRC {
		return crcFlowID(fsName, fDistinguisher)
	}
	return shaFlowID(fsName, fDistinguisher)
}

func shaFlowID(fsName, fDistinguisher string) uint64 {
	hash := sha256.New()
	var sep = [1]byte{0}
	hash.Write([]byte(fsName))
	hash.Write(sep[:])
	hash.Write([]byte(fDistinguisher))
	var sum [32]byte
	hash.Sum(sum[:0])
	return binary.LittleEndian.Uint64(sum[:8])
}

var crcHashTable = crc64.MakeTable(crc64.ECMA)

func crcFlowID(fsName, fDistinguisher string) uint64 {
	var hash uint64
	hash = crc64.Update(hash, crcHashTable, []byte(fsName))
	hash = crc64.Update(hash, crcHashTable, []byte{1})
	hash = crc64.Update(hash, crcHashTable, []byte(fDistinguisher))
	return hash
}
