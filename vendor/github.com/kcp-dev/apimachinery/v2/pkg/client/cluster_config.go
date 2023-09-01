/*
Copyright 2022 The KCP Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package client

import (
	"github.com/kcp-dev/logicalcluster/v3"

	"k8s.io/client-go/rest"
)

// SetCluster modifies the config host path to include the
// cluster endpoint.
//
// Note: it is the caller responsibility to make a copy of the rest config.
func SetCluster(cfg *rest.Config, clusterPath logicalcluster.Path) *rest.Config {
	cfg.Host += clusterPath.RequestPath()
	return cfg
}
