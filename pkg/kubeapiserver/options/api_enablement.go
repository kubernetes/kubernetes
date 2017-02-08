/*
Copyright 2017 The Kubernetes Authors.

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

	utilflag "k8s.io/apiserver/pkg/util/flag"
)

// APIEnablementOptions contains the options for which resources to turn on and off.
// Given small aggregated API servers, this option isn't required for "normal" API servers
type APIEnablementOptions struct {
	RuntimeConfig utilflag.ConfigurationMap
}

func NewAPIEnablementOptions() *APIEnablementOptions {
	return &APIEnablementOptions{
		RuntimeConfig: make(utilflag.ConfigurationMap),
	}
}

// AddFlags adds flags for a specific APIServer to the specified FlagSet
func (s *APIEnablementOptions) AddFlags(fs *pflag.FlagSet) {
	fs.Var(&s.RuntimeConfig, "runtime-config", ""+
		"A set of key=value pairs that describe runtime configuration that may be passed "+
		"to apiserver. apis/<groupVersion> key can be used to turn on/off specific api versions. "+
		"apis/<groupVersion>/<resource> can be used to turn on/off specific resources. api/all and "+
		"api/legacy are special keys to control all and legacy api versions respectively.")
}
