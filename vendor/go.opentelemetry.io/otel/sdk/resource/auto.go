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
)

var (
	// ErrPartialResource is returned by a detector when complete source
	// information for a Resource is unavailable or the source information
	// contains invalid values that are omitted from the returned Resource.
	ErrPartialResource = errors.New("partial resource")
)

// Detector detects OpenTelemetry resource information
type Detector interface {
	// Detect returns an initialized Resource based on gathered information.
	// If the source information to construct a Resource contains invalid
	// values, a Resource is returned with the valid parts of the source
	// information used for initialization along with an appropriately
	// wrapped ErrPartialResource error.
	Detect(ctx context.Context) (*Resource, error)
}

// Detect calls all input detectors sequentially and merges each result with the previous one.
// It returns the merged error too.
func Detect(ctx context.Context, detectors ...Detector) (*Resource, error) {
	var autoDetectedRes *Resource
	var errInfo []string
	for _, detector := range detectors {
		if detector == nil {
			continue
		}
		res, err := detector.Detect(ctx)
		if err != nil {
			errInfo = append(errInfo, err.Error())
			if !errors.Is(err, ErrPartialResource) {
				continue
			}
		}
		autoDetectedRes = Merge(autoDetectedRes, res)
	}

	var aggregatedError error
	if len(errInfo) > 0 {
		aggregatedError = fmt.Errorf("detecting resources: %s", errInfo)
	}
	return autoDetectedRes, aggregatedError
}
