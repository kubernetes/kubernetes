/*
 *
 * Copyright 2020 gRPC authors.
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

package clusterimpl

import (
	"encoding/json"

	internalserviceconfig "google.golang.org/grpc/internal/serviceconfig"
	"google.golang.org/grpc/internal/xds/bootstrap"
	"google.golang.org/grpc/serviceconfig"
)

// DropConfig contains the category, and drop ratio.
type DropConfig struct {
	Category           string
	RequestsPerMillion uint32
}

// LBConfig is the balancer config for cluster_impl balancer.
type LBConfig struct {
	serviceconfig.LoadBalancingConfig `json:"-"`

	Cluster        string `json:"cluster,omitempty"`
	EDSServiceName string `json:"edsServiceName,omitempty"`
	// LoadReportingServer is the LRS server to send load reports to. If not
	// present, load reporting will be disabled.
	LoadReportingServer   *bootstrap.ServerConfig `json:"lrsLoadReportingServer,omitempty"`
	MaxConcurrentRequests *uint32                 `json:"maxConcurrentRequests,omitempty"`
	DropCategories        []DropConfig            `json:"dropCategories,omitempty"`
	// TelemetryLabels are the telemetry Labels associated with this cluster.
	TelemetryLabels map[string]string                     `json:"telemetryLabels,omitempty"`
	ChildPolicy     *internalserviceconfig.BalancerConfig `json:"childPolicy,omitempty"`
}

func parseConfig(c json.RawMessage) (*LBConfig, error) {
	var cfg LBConfig
	if err := json.Unmarshal(c, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}
