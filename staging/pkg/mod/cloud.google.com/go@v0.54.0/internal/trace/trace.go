// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package trace

import (
	"context"
	"fmt"

	"go.opencensus.io/trace"
	"google.golang.org/api/googleapi"
	"google.golang.org/genproto/googleapis/rpc/code"
	"google.golang.org/grpc/status"
)

// StartSpan adds a span to the trace with the given name.
func StartSpan(ctx context.Context, name string) context.Context {
	ctx, _ = trace.StartSpan(ctx, name)
	return ctx
}

// EndSpan ends a span with the given error.
func EndSpan(ctx context.Context, err error) {
	span := trace.FromContext(ctx)
	if err != nil {
		span.SetStatus(toStatus(err))
	}
	span.End()
}

// toStatus interrogates an error and converts it to an appropriate
// OpenCensus status.
func toStatus(err error) trace.Status {
	if err2, ok := err.(*googleapi.Error); ok {
		return trace.Status{Code: httpStatusCodeToOCCode(err2.Code), Message: err2.Message}
	} else if s, ok := status.FromError(err); ok {
		return trace.Status{Code: int32(s.Code()), Message: s.Message()}
	} else {
		return trace.Status{Code: int32(code.Code_UNKNOWN), Message: err.Error()}
	}
}

// TODO(deklerk): switch to using OpenCensus function when it becomes available.
// Reference: https://github.com/googleapis/googleapis/blob/26b634d2724ac5dd30ae0b0cbfb01f07f2e4050e/google/rpc/code.proto
func httpStatusCodeToOCCode(httpStatusCode int) int32 {
	switch httpStatusCode {
	case 200:
		return int32(code.Code_OK)
	case 499:
		return int32(code.Code_CANCELLED)
	case 500:
		return int32(code.Code_UNKNOWN) // Could also be Code_INTERNAL, Code_DATA_LOSS
	case 400:
		return int32(code.Code_INVALID_ARGUMENT) // Could also be Code_OUT_OF_RANGE
	case 504:
		return int32(code.Code_DEADLINE_EXCEEDED)
	case 404:
		return int32(code.Code_NOT_FOUND)
	case 409:
		return int32(code.Code_ALREADY_EXISTS) // Could also be Code_ABORTED
	case 403:
		return int32(code.Code_PERMISSION_DENIED)
	case 401:
		return int32(code.Code_UNAUTHENTICATED)
	case 429:
		return int32(code.Code_RESOURCE_EXHAUSTED)
	case 501:
		return int32(code.Code_UNIMPLEMENTED)
	case 503:
		return int32(code.Code_UNAVAILABLE)
	default:
		return int32(code.Code_UNKNOWN)
	}
}

// TODO: (odeke-em): perhaps just pass around spans due to the cost
// incurred from using trace.FromContext(ctx) yet we could avoid
// throwing away the work done by ctx, span := trace.StartSpan.
func TracePrintf(ctx context.Context, attrMap map[string]interface{}, format string, args ...interface{}) {
	var attrs []trace.Attribute
	for k, v := range attrMap {
		var a trace.Attribute
		switch v := v.(type) {
		case string:
			a = trace.StringAttribute(k, v)
		case bool:
			a = trace.BoolAttribute(k, v)
		case int:
			a = trace.Int64Attribute(k, int64(v))
		case int64:
			a = trace.Int64Attribute(k, v)
		default:
			a = trace.StringAttribute(k, fmt.Sprintf("%#v", v))
		}
		attrs = append(attrs, a)
	}
	trace.FromContext(ctx).Annotatef(attrs, format, args...)
}
