/*
Copyright The Kubernetes Authors.

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

package affinity

import (
	"errors"
	"fmt"
	"strings"
)

var (
	ErrClaimSelfInconsistent = errors.New("claim self-inconsistent")
	ErrSliceKeyCollision     = errors.New("slice key collision")
	ErrCELEval               = errors.New("cel evaluation failed")
	ErrNonStringReturn       = errors.New("cel expression returned non-string")
	ErrCostExceeded          = errors.New("cel evaluation cost exceeded")
	ErrMissingField          = errors.New("missing field")

	errNonStringResult   = errors.New("non-string result")
	errCostLimitExceeded = errors.New("cost limit exceeded")
)

type Error struct {
	Kind    error
	Pool    PoolKey
	Request string
	Device  string
	// ExtractorIndex is the index of the sharingAffinity extractor, or -1 when
	// the error is not specific to a single extractor.
	ExtractorIndex int
	Key            string
	Expression     string
	Err            error
}

func (e *Error) Error() string {
	if e == nil {
		return "<nil>"
	}
	var b strings.Builder
	if e.Kind != nil {
		b.WriteString(e.Kind.Error())
	} else {
		b.WriteString("sharing affinity extraction failed")
	}
	if e.Pool.Driver != "" || e.Pool.Name != "" {
		fmt.Fprintf(&b, " for pool %q/%q generation %d", e.Pool.Driver, e.Pool.Name, e.Pool.Generation)
	}
	if e.Request != "" {
		fmt.Fprintf(&b, ", request %q", e.Request)
	}
	if e.Device != "" {
		fmt.Fprintf(&b, ", device %q", e.Device)
	}
	if e.ExtractorIndex >= 0 {
		fmt.Fprintf(&b, ", extractor %d", e.ExtractorIndex)
	}
	if e.Key != "" {
		fmt.Fprintf(&b, ", key %q", e.Key)
	}
	if e.Err != nil {
		fmt.Fprintf(&b, ": %v", e.Err)
	}
	return b.String()
}

func (e *Error) Unwrap() error {
	return e.Err
}

func (e *Error) Is(target error) bool {
	return e != nil && errors.Is(e.Kind, target)
}
