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

package internal // import "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc/internal"

import (
	"strings"

	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

// ParseFullMethod returns a span name following the OpenTelemetry semantic
// conventions as well as all applicable span attribute.KeyValue attributes based
// on a gRPC's FullMethod.
//
// Parsing is consistent with grpc-go implementation:
// https://github.com/grpc/grpc-go/blob/v1.57.0/internal/grpcutil/method.go#L26-L39
func ParseFullMethod(fullMethod string) (string, []attribute.KeyValue) {
	if !strings.HasPrefix(fullMethod, "/") {
		// Invalid format, does not follow `/package.service/method`.
		return fullMethod, nil
	}
	name := fullMethod[1:]
	pos := strings.LastIndex(name, "/")
	if pos < 0 {
		// Invalid format, does not follow `/package.service/method`.
		return name, nil
	}
	service, method := name[:pos], name[pos+1:]

	var attrs []attribute.KeyValue
	if service != "" {
		attrs = append(attrs, semconv.RPCService(service))
	}
	if method != "" {
		attrs = append(attrs, semconv.RPCMethod(method))
	}
	return name, attrs
}
