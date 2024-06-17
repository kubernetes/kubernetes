// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// ErrPartialResource is returned by a detector when complete source
// information for a Resource is unavailable or the source information
// contains invalid values that are omitted from the returned Resource.
var ErrPartialResource = errors.New("partial resource")

// Detector detects OpenTelemetry resource information.
type Detector interface {
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Detect returns an initialized Resource based on gathered information.
	// If the source information to construct a Resource contains invalid
	// values, a Resource is returned with the valid parts of the source
	// information used for initialization along with an appropriately
	// wrapped ErrPartialResource error.
	Detect(ctx context.Context) (*Resource, error)
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.
}

// Detect calls all input detectors sequentially and merges each result with the previous one.
// It returns the merged error too.
func Detect(ctx context.Context, detectors ...Detector) (*Resource, error) {
	r := new(Resource)
	return r, detect(ctx, r, detectors)
}

// detect runs all detectors using ctx and merges the result into res. This
// assumes res is allocated and not nil, it will panic otherwise.
func detect(ctx context.Context, res *Resource, detectors []Detector) error {
	var (
		r    *Resource
		errs detectErrs
		err  error
	)

	for _, detector := range detectors {
		if detector == nil {
			continue
		}
		r, err = detector.Detect(ctx)
		if err != nil {
			errs = append(errs, err)
			if !errors.Is(err, ErrPartialResource) {
				continue
			}
		}
		r, err = Merge(res, r)
		if err != nil {
			errs = append(errs, err)
		}
		*res = *r
	}

	if len(errs) == 0 {
		return nil
	}
	return errs
}

type detectErrs []error

func (e detectErrs) Error() string {
	errStr := make([]string, len(e))
	for i, err := range e {
		errStr[i] = fmt.Sprintf("* %s", err)
	}

	format := "%d errors occurred detecting resource:\n\t%s"
	return fmt.Sprintf(format, len(e), strings.Join(errStr, "\n\t"))
}

func (e detectErrs) Unwrap() error {
	switch len(e) {
	case 0:
		return nil
	case 1:
		return e[0]
	}
	return e[1:]
}

func (e detectErrs) Is(target error) bool {
	return len(e) != 0 && errors.Is(e[0], target)
}
