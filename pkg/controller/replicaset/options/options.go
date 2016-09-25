/*
Copyright 2016 The Kubernetes Authors.

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
	"github.com/spf13/pflag"
)

type ReplicasetControllerOptions struct {
	ConcurrentRSSyncs int
}

func NewReplicasetControllerOptions() ReplicasetControllerOptions {
	return ReplicasetControllerOptions{
		ConcurrentRSSyncs: 5,
	}
}

func (o *ReplicasetControllerOptions) AddFlags(fs *pflag.FlagSet) {
	fs.IntVar(&o.ConcurrentRSSyncs, "concurrent-replicaset-syncs", o.ConcurrentRSSyncs, "The number of replicasets that are allowed to sync concurrently. Larger number = more reponsive replica management, but more CPU (and network) load")
}
