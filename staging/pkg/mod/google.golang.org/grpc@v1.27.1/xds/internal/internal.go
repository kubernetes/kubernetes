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

package internal

import (
	"fmt"

	corepb "github.com/envoyproxy/go-control-plane/envoy/api/v2/core"
)

type clientID string

// XDSClientID is the attributes key used to pass the address of the xdsClient
// object shared between the resolver and the balancer. The xdsClient object is
// created by the resolver and passed to the balancer.
const XDSClientID = clientID("xdsClientID")

// Locality is xds.Locality without XXX fields, so it can be used as map
// keys.
//
// xds.Locality cannot be map keys because one of the XXX fields is a slice.
//
// This struct should only be used as map keys. Use the proto message directly
// in all other places.
//
// TODO: rename to LocalityID.
type Locality struct {
	Region  string
	Zone    string
	SubZone string
}

func (lamk Locality) String() string {
	return fmt.Sprintf("%s-%s-%s", lamk.Region, lamk.Zone, lamk.SubZone)
}

// ToProto convert Locality to the proto representation.
func (lamk Locality) ToProto() *corepb.Locality {
	return &corepb.Locality{
		Region:  lamk.Region,
		Zone:    lamk.Zone,
		SubZone: lamk.SubZone,
	}
}
