/*
Copyright 2026 The Kubernetes Authors.

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

package conntrack

import (
	"context"
	"fmt"
	"k8s.io/klog/v2"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
)

func getConntrackMax(ctx context.Context, config *kubeproxyconfig.KubeProxyConntrackConfiguration, numCPU int) (int, error) {
	logger := klog.FromContext(ctx)
	if config.MaxPerCore != nil && *config.MaxPerCore > 0 {
		floor := 0
		if config.Min != nil {
			floor = int(*config.Min)
		}
		limit := max(int64(*config.MaxPerCore)*int64(numCPU), int64(floor))
		maxLimit := int64(1048576)
		if config.Max != nil {
			maxLimit = int64(*config.Max)
		}
		if maxLimit > 0 && limit > maxLimit {
			logger.V(3).Info("GetConntrackMax: capping calculated conntrack limit", "calculated", limit, "limit", maxLimit)
			limit = maxLimit
		}
		if limit > int64(int(^uint(0)>>1)) {
			return 0, fmt.Errorf("calculated conntrack limit %d exceeds the maximum integer value on this platform", limit)
		}
		return int(limit), nil
	}
	return 0, nil
}
