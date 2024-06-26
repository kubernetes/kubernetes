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

package semconvutil // import "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp/internal/semconvutil"

// Generate semconvutil package:
//go:generate gotmpl --body=../../../../../../internal/shared/semconvutil/httpconv_test.go.tmpl "--data={}" --out=httpconv_test.go
//go:generate gotmpl --body=../../../../../../internal/shared/semconvutil/httpconv.go.tmpl "--data={}" --out=httpconv.go
//go:generate gotmpl --body=../../../../../../internal/shared/semconvutil/netconv_test.go.tmpl "--data={}" --out=netconv_test.go
//go:generate gotmpl --body=../../../../../../internal/shared/semconvutil/netconv.go.tmpl "--data={}" --out=netconv.go
