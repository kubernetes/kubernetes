/*
Copyright 2023 The Kubernetes Authors.

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

package prober

import (
	"context"
	"fmt"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/probe"
)

// unknownProbeTypeError is used by dummyProbeRunner for returning error.
var unknownProbeTypeError = fmt.Errorf("unknown probe type")

// unknownProbeError is used by dummyProbeRunner for returning error.
var unknownProbeHandlerError = fmt.Errorf("missing probe handler")

// dummyProbeRunner covers three corner/hypothetical cases:
// 1. Unknown Probe Type
// 2. Nil Probe Spec
// 3. Unknown Probe Handler
type dummyProbeRunner struct {
	isProbeTypeUnknown bool
	isProbeNil         bool
	isHandlerUnknown   bool
}

// newDummyProbeRunner returns dummyProbeRunner which implements probeRunner.
func newDummyProbeRunner(isProbeTypeUnknown, isProbeNil, isHandlerUnknown bool) *dummyProbeRunner {
	return &dummyProbeRunner{
		isProbeTypeUnknown: isProbeTypeUnknown, isProbeNil: isProbeNil, isHandlerUnknown: isHandlerUnknown}
}

func (dp *dummyProbeRunner) sync(_ v1.Container, _ v1.PodStatus, _ probeType) error {
	return nil
}

func (dp *dummyProbeRunner) run(_ context.Context, _ v1.Container, _ v1.PodStatus, _ probeType, _ *prober) (probe.Result, string, error) {
	// case 1: unknown probe type
	if dp.isProbeTypeUnknown {
		return probe.Failure, "", unknownProbeTypeError
	}

	// case 2: probe is nil
	if dp.isProbeNil {
		return probe.Success, "", nil
	}

	// case 3: unknown handler / empty probe
	if dp.isHandlerUnknown {
		return probe.Unknown, "", unknownProbeHandlerError
	}

	// default: returning true as we have covered all the corner cases.
	return probe.Success, "", nil

}
