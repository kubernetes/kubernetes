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

type ServiceControllerOptions struct {
	ServiceSyncPeriod time.Duration
}

func NewServiceControllerOptions() ServiceControllerOptions {
	return ServiceControllerOptions{
		ServiceSyncPeriod: 5 * time.Minute,
	}
}

func (o *ServiceControllerOptions) AddFlags(fs *pflag.FlagSet) {
	fs.DurationVar(&o.ServiceSyncPeriod, "service-sync-period", o.ServiceSyncPeriod, "The period for syncing services with their external load balancers")
}
