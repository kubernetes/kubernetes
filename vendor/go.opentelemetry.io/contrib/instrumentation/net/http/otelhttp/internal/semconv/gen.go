// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package semconv // import "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp/internal/semconv"

// Generate semconv package:
//go:generate gotmpl --body=../../../../../../internal/shared/semconv/bench_test.go.tmpl "--data={ \"pkg\": \"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp\" }" --out=bench_test.go
//go:generate gotmpl --body=../../../../../../internal/shared/semconv/common_test.go.tmpl "--data={ \"pkg\": \"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp\" }" --out=common_test.go
//go:generate gotmpl --body=../../../../../../internal/shared/semconv/server.go.tmpl "--data={ \"pkg\": \"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp\" }" --out=server.go
//go:generate gotmpl --body=../../../../../../internal/shared/semconv/server_test.go.tmpl "--data={ \"pkg\": \"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp\" }" --out=server_test.go
//go:generate gotmpl --body=../../../../../../internal/shared/semconv/client.go.tmpl "--data={ \"pkg\": \"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp\" }" --out=client.go
//go:generate gotmpl --body=../../../../../../internal/shared/semconv/client_test.go.tmpl "--data={ \"pkg\": \"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp\" }" --out=client_test.go
//go:generate gotmpl --body=../../../../../../internal/shared/semconv/httpconvtest_test.go.tmpl "--data={ \"pkg\": \"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp\" }" --out=httpconvtest_test.go
//go:generate gotmpl --body=../../../../../../internal/shared/semconv/util.go.tmpl "--data={ \"pkg\": \"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp\" }" --out=util.go
//go:generate gotmpl --body=../../../../../../internal/shared/semconv/util_test.go.tmpl "--data={ \"pkg\": \"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp\" }" --out=util_test.go
