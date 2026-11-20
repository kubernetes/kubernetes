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

package storage

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/testing/correctness"
)

type errorCategory string

const (
	errorNotFound        errorCategory = "notFound"
	errorAlreadyExists   errorCategory = "alreadyExists"
	errorConflicted      errorCategory = "conflicted"
	errorRejected        errorCategory = "rejected"
	errorBadRequest      errorCategory = "badRequest"
	errorTooOld          errorCategory = "tooOld"
	errorTooLargeRV      errorCategory = "tooLargeRV"
	errorTooManyRequests errorCategory = "tooManyRequests"
	errorTimeout         errorCategory = "timeout"
	errorUnreachable     errorCategory = "unreachable"
	errorCorrupt         errorCategory = "corrupt"
	errorCanceled        errorCategory = "canceled"
	errorOther           errorCategory = "other"
)

type operationStats struct {
	executed  int
	succeeded int
	errors    map[errorCategory]int
}

func (s *operationStats) record(err error) {
	s.executed++
	if err == nil {
		s.succeeded++
		return
	}
	if s.errors == nil {
		s.errors = make(map[errorCategory]int)
	}
	s.errors[classifyError(err)]++
}

func (s operationStats) String() string {
	failed := s.executed - s.succeeded
	if failed == 0 {
		return fmt.Sprintf("executed=%d succeeded=%d failed=0", s.executed, s.succeeded)
	}
	parts := make([]string, 0, len(s.errors))
	for _, cat := range slices.Sorted(maps.Keys(s.errors)) {
		parts = append(parts, fmt.Sprintf("%s=%d", cat, s.errors[cat]))
	}
	return fmt.Sprintf("executed=%d succeeded=%d failed=%d (%s)", s.executed, s.succeeded, failed, strings.Join(parts, ", "))
}

func classifyError(err error) errorCategory {
	switch {
	case errors.Is(err, errDeleteRejected), errors.Is(err, errUpdateRejected):
		return errorRejected
	case storage.IsNotFound(err), apierrors.IsNotFound(err):
		return errorNotFound
	case storage.IsExist(err), apierrors.IsAlreadyExists(err):
		return errorAlreadyExists
	case storage.IsConflict(err), storage.IsInvalidObj(err), apierrors.IsConflict(err):
		return errorConflicted
	case apierrors.IsBadRequest(err), storage.IsInvalidError(err), apierrors.IsInvalid(err):
		return errorBadRequest
	case apierrors.IsResourceExpired(err):
		return errorTooOld
	case storage.IsTooLargeResourceVersion(err):
		return errorTooLargeRV
	case apierrors.IsTooManyRequests(err):
		return errorTooManyRequests
	case storage.IsRequestTimeout(err), apierrors.IsTimeout(err), apierrors.IsServerTimeout(err):
		return errorTimeout
	case storage.IsUnreachable(err):
		return errorUnreachable
	case storage.IsCorruptObject(err):
		return errorCorrupt
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return errorCanceled
	default:
		return errorOther
	}
}

func summarizeUnaryOperations(ops []correctness.Operation) (operationStats, map[correctness.OpType]*operationStats) {
	var total operationStats
	byOp := make(map[correctness.OpType]*operationStats)
	for _, op := range ops {
		total.record(op.Response.Err)
		s := byOp[op.Request.Op]
		if s == nil {
			s = &operationStats{}
			byOp[op.Request.Op] = s
		}
		s.record(op.Response.Err)
	}
	return total, byOp
}

func summarizeWatchOperations(watches []correctness.WatchOperation) (operationStats, int) {
	var total operationStats
	events := 0
	for _, w := range watches {
		events += len(w.Response.Events)
		total.record(watchError(w))
	}
	return total, events
}

func watchError(w correctness.WatchOperation) error {
	if w.Response.Err != nil {
		return w.Response.Err
	}
	if n := len(w.Response.Events); n > 0 && w.Response.Events[n-1].Type == watch.Error {
		return apierrors.FromObject(w.Response.Events[n-1].Object)
	}
	return nil
}
