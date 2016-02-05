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
	"github.com/spf13/pflag"
)

type GarbageCollectorOptions struct {
	TerminatedPodGCThreshold int
}

func NewGarbageCollectorOptions() GarbageCollectorOptions {
	return GarbageCollectorOptions{
		TerminatedPodGCThreshold: 12500,
	}
}

func (o *GarbageCollectorOptions) AddFlags(fs *pflag.FlagSet) {
	fs.IntVar(&o.TerminatedPodGCThreshold, "terminated-pod-gc-threshold", o.TerminatedPodGCThreshold,
		"Number of terminated pods that can exist before the terminated pod garbage collector starts deleting terminated pods. "+
			"If <= 0, the terminated pod garbage collector is disabled.")
}
