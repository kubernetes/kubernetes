/*
Copyright 2022 The Kubernetes Authors.

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

package klogflags

import (
	"flag"

	"k8s.io/klog/v2"
)

// Init is a replacement for klog.InitFlags which only adds those flags
// that are still supported for Kubernetes components (i.e. -v and -vmodule).
// See
// https://github.com/kubernetes/enhancements/tree/master/keps/sig-instrumentation/2845-deprecate-klog-specific-flags-in-k8s-components.
func Init(fs *flag.FlagSet) {
	var allFlags flag.FlagSet
	klog.InitFlags(&allFlags)
	if fs == nil {
		fs = flag.CommandLine
	}
	allFlags.VisitAll(func(f *flag.Flag) {
		switch f.Name {
		case "v", "vmodule":
			fs.Var(f.Value, f.Name, f.Usage)
		}
	})
}
