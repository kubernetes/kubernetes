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

package flagutil

import (
	"flag"

	// kubelet attempts to customize default values for some cadvisor flags, so
	// make sure that we pick these up.
	_ "k8s.io/kubernetes/pkg/kubelet/cadvisor"
)

// FlagFunc retrieves a specific flag instance; returns nil if the flag is not configured.
type FlagFunc func() *flag.Flag

// NameValue returns the name and value of a flag, if it exists, otherwise empty strings.
func (ff FlagFunc) NameValue() (name, value string) {
	if f := ff(); f != nil {
		name, value = f.Name, f.Value.String()
	}
	return
}

func flagFunc(name string) FlagFunc { return func() *flag.Flag { return flag.Lookup(name) } }

// Cadvisor fields return the configured values of cadvisor global flags
var Cadvisor = struct {
	HousekeepingInterval       FlagFunc
	GlobalHousekeepingInterval FlagFunc
}{
	flagFunc("housekeeping_interval"),
	flagFunc("global_housekeeping_interval"),
}
