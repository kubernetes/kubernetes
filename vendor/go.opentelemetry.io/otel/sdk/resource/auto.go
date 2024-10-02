// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

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

// Detect returns a new [Resource] merged from all the Resources each of the
// detectors produces. Each of the detectors are called sequentially, in the
// order they are passed, merging the produced resource into the previous.
//
// This may return a partial Resource along with an error containing
// [ErrPartialResource] if that error is returned from a detector. It may also
// return a merge-conflicting Resource along with an error containing
// [ErrSchemaURLConflict] if merging Resources from different detectors results
// in a schema URL conflict. It is up to the caller to determine if this
// returned Resource should be used or not.
//
// If one of the detectors returns an error that is not [ErrPartialResource],
// the resource produced by the detector will not be merged and the returned
// error will wrap that detector's error.
func Detect(ctx context.Context, detectors ...Detector) (*Resource, error) {
	r := new(Resource)
	return r, detect(ctx, r, detectors)
}

// detect runs all detectors using ctx and merges the result into res. This
// assumes res is allocated and not nil, it will panic otherwise.
//
// If the detectors or merging resources produces any errors (i.e.
// [ErrPartialResource] [ErrSchemaURLConflict]), a single error wrapping all of
// these errors will be returned. Otherwise, nil is returned.
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
	if errors.Is(errs, ErrSchemaURLConflict) {
		// If there has been a merge conflict, ensure the resource has no
		// schema URL.
		res.schemaURL = ""
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
