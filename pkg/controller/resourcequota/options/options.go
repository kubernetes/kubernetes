/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package options

import (
	"time"

	"github.com/spf13/pflag"
)

type ResourceQuotaControllerOptions struct {
	ConcurrentResourceQuotaSyncs int
	ResourceQuotaSyncPeriod      time.Duration
}

func NewResourceQuotaControllerOptions() ResourceQuotaControllerOptions {
	return ResourceQuotaControllerOptions{
		ConcurrentResourceQuotaSyncs: 5,
		ResourceQuotaSyncPeriod:      5 * time.Minute,
	}
}

func (o *ResourceQuotaControllerOptions) AddFlags(fs *pflag.FlagSet) {
	fs.IntVar(&o.ConcurrentResourceQuotaSyncs, "concurrent-resource-quota-syncs", o.ConcurrentResourceQuotaSyncs, "The number of resource quotas that are allowed to sync concurrently. Larger number = more responsive quota management, but more CPU (and network) load")
	fs.DurationVar(&o.ResourceQuotaSyncPeriod, "resource-quota-sync-period", o.ResourceQuotaSyncPeriod, "The period for syncing quota usage status in the system")

}
