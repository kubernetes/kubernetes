/*
 *
 * Copyright 2019 gRPC authors.
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
 */

package edsbalancer

import "google.golang.org/grpc/balancer"

// The old xds balancer implements logic for both CDS and EDS. With the new
// design, CDS is split and moved to a separate balancer, and the xds balancer
// becomes the EDS balancer.
//
// To keep the existing tests working, this file regisger EDS balancer under the
// old xds balancer name.
//
// TODO: delete this file when migration to new workflow (LDS, RDS, CDS, EDS) is
// done.

const xdsName = "xds_experimental"

func init() {
	balancer.Register(&xdsBalancerBuilder{})
}

// xdsBalancerBuilder register edsBalancerBuilder (now with name
// "experimental_eds") under the old name "xds_experimental".
type xdsBalancerBuilder struct {
	edsBalancerBuilder
}

func (b *xdsBalancerBuilder) Name() string {
	return xdsName
}
