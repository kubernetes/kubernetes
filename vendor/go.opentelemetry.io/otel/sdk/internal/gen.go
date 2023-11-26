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

package internal // import "go.opentelemetry.io/otel/sdk/internal"

//go:generate gotmpl --body=../../internal/shared/matchers/expectation.go.tmpl "--data={}" --out=matchers/expectation.go
//go:generate gotmpl --body=../../internal/shared/matchers/expecter.go.tmpl "--data={}" --out=matchers/expecter.go
//go:generate gotmpl --body=../../internal/shared/matchers/temporal_matcher.go.tmpl "--data={}" --out=matchers/temporal_matcher.go

//go:generate gotmpl --body=../../internal/shared/internaltest/alignment.go.tmpl "--data={}" --out=internaltest/alignment.go
//go:generate gotmpl --body=../../internal/shared/internaltest/env.go.tmpl "--data={}" --out=internaltest/env.go
//go:generate gotmpl --body=../../internal/shared/internaltest/env_test.go.tmpl "--data={}" --out=internaltest/env_test.go
//go:generate gotmpl --body=../../internal/shared/internaltest/errors.go.tmpl "--data={}" --out=internaltest/errors.go
//go:generate gotmpl --body=../../internal/shared/internaltest/harness.go.tmpl "--data={\"matchersImportPath\": \"go.opentelemetry.io/otel/sdk/internal/matchers\"}" --out=internaltest/harness.go
//go:generate gotmpl --body=../../internal/shared/internaltest/text_map_carrier.go.tmpl "--data={}" --out=internaltest/text_map_carrier.go
//go:generate gotmpl --body=../../internal/shared/internaltest/text_map_carrier_test.go.tmpl "--data={}" --out=internaltest/text_map_carrier_test.go
//go:generate gotmpl --body=../../internal/shared/internaltest/text_map_propagator.go.tmpl "--data={}" --out=internaltest/text_map_propagator.go
//go:generate gotmpl --body=../../internal/shared/internaltest/text_map_propagator_test.go.tmpl "--data={}" --out=internaltest/text_map_propagator_test.go
