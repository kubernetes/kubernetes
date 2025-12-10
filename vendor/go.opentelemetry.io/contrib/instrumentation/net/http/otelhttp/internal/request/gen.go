// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package request // import "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp/internal/request"

// Generate request package:
//go:generate gotmpl --body=../../../../../../internal/shared/request/body_wrapper.go.tmpl "--data={}" --out=body_wrapper.go
//go:generate gotmpl --body=../../../../../../internal/shared/request/body_wrapper_test.go.tmpl "--data={}" --out=body_wrapper_test.go
//go:generate gotmpl --body=../../../../../../internal/shared/request/resp_writer_wrapper.go.tmpl "--data={}" --out=resp_writer_wrapper.go
//go:generate gotmpl --body=../../../../../../internal/shared/request/resp_writer_wrapper_test.go.tmpl "--data={}" --out=resp_writer_wrapper_test.go
