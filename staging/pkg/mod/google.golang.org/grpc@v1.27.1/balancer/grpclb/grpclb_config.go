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
 *
 */

package grpclb

import (
	"encoding/json"

	"google.golang.org/grpc"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/serviceconfig"
)

const (
	roundRobinName = roundrobin.Name
	pickFirstName  = grpc.PickFirstBalancerName
)

type grpclbServiceConfig struct {
	serviceconfig.LoadBalancingConfig
	ChildPolicy *[]map[string]json.RawMessage
}

func (b *lbBuilder) ParseConfig(lbConfig json.RawMessage) (serviceconfig.LoadBalancingConfig, error) {
	ret := &grpclbServiceConfig{}
	if err := json.Unmarshal(lbConfig, ret); err != nil {
		return nil, err
	}
	return ret, nil
}

func childIsPickFirst(sc *grpclbServiceConfig) bool {
	if sc == nil {
		return false
	}
	childConfigs := sc.ChildPolicy
	if childConfigs == nil {
		return false
	}
	for _, childC := range *childConfigs {
		// If round_robin exists before pick_first, return false
		if _, ok := childC[roundRobinName]; ok {
			return false
		}
		// If pick_first is before round_robin, return true
		if _, ok := childC[pickFirstName]; ok {
			return true
		}
	}
	return false
}
