/*
 *
 * Copyright 2023 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package weightedroundrobin

import (
	iserviceconfig "google.golang.org/grpc/internal/serviceconfig"
	"google.golang.org/grpc/serviceconfig"
)

type lbConfig struct {
	serviceconfig.LoadBalancingConfig `json:"-"`

	// Whether to enable out-of-band utilization reporting collection from the
	// endpoints.  By default, per-request utilization reporting is used.
	EnableOOBLoadReport bool `json:"enableOobLoadReport,omitempty"`

	// Load reporting interval to request from the server.  Note that the
	// server may not provide reports as frequently as the client requests.
	// Used only when enable_oob_load_report is true.  Default is 10 seconds.
	OOBReportingPeriod iserviceconfig.Duration `json:"oobReportingPeriod,omitempty"`

	// A given endpoint must report load metrics continuously for at least this
	// long before the endpoint weight will be used.  This avoids churn when
	// the set of endpoint addresses changes.  Takes effect both immediately
	// after we establish a connection to an endpoint and after
	// weight_expiration_period has caused us to stop using the most recent
	// load metrics.  Default is 10 seconds.
	BlackoutPeriod iserviceconfig.Duration `json:"blackoutPeriod,omitempty"`

	// If a given endpoint has not reported load metrics in this long,
	// then we stop using the reported weight.  This ensures that we do
	// not continue to use very stale weights.  Once we stop using a stale
	// value, if we later start seeing fresh reports again, the
	// blackout_period applies.  Defaults to 3 minutes.
	WeightExpirationPeriod iserviceconfig.Duration `json:"weightExpirationPeriod,omitempty"`

	// How often endpoint weights are recalculated.  Default is 1 second.
	WeightUpdatePeriod iserviceconfig.Duration `json:"weightUpdatePeriod,omitempty"`

	// The multiplier used to adjust endpoint weights with the error rate
	// calculated as eps/qps. Default is 1.0.
	ErrorUtilizationPenalty float64 `json:"errorUtilizationPenalty,omitempty"`
}
