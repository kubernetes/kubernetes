// +build windows

/*
Copyright 2021 The Kubernetes Authors.

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

func (o *Options) addOSFlags(fs *pflag.FlagSet) {
	fs.BoolVar(&o.WindowsService, "windows-service", o.WindowsService, "Enable Windows Service Control Manager API integration")
	fs.StringVar(&o.ComponentConfig.Winkernel.SourceVip, "source-vip", o.ComponentConfig.Winkernel.SourceVip, "The IP address of the source VIP for non-DSR.")
	fs.StringVar(&o.ComponentConfig.Winkernel.NetworkName, "network-name", o.ComponentConfig.Winkernel.NetworkName, "The name of the cluster network.")
	fs.BoolVar(&o.ComponentConfig.Winkernel.EnableDSR, "enable-dsr", o.ComponentConfig.Winkernel.EnableDSR, "If true make kube-proxy apply DSR policies for service VIP")
}
