/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package flags

import (
	"flag"
	"fmt"

	"golang.org/x/net/context"

	"github.com/vmware/govmomi/vim25/types"
)

type HostConnectFlag struct {
	common

	types.HostConnectSpec

	noverify bool
}

var hostConnectFlagKey = flagKey("hostConnect")

func NewHostConnectFlag(ctx context.Context) (*HostConnectFlag, context.Context) {
	if v := ctx.Value(hostConnectFlagKey); v != nil {
		return v.(*HostConnectFlag), ctx
	}

	v := &HostConnectFlag{}
	ctx = context.WithValue(ctx, hostConnectFlagKey, v)
	return v, ctx
}

func (flag *HostConnectFlag) Register(ctx context.Context, f *flag.FlagSet) {
	flag.RegisterOnce(func() {
		f.StringVar(&flag.HostName, "hostname", "", "Hostname or IP address of the host")
		f.StringVar(&flag.UserName, "username", "", "Username of administration account on the host")
		f.StringVar(&flag.Password, "password", "", "Password of administration account on the host")
		f.StringVar(&flag.SslThumbprint, "fingerprint", "", "Fingerprint of the host's SSL certificate")
		f.BoolVar(&flag.Force, "force", false, "Force when host is managed by another VC")

		f.BoolVar(&flag.noverify, "noverify", false, "When true, ignore host SSL certificate verification error")
	})
}

func (flag *HostConnectFlag) Process(ctx context.Context) error {
	return nil
}

// AcceptThumbprint returns nil if the given error is an SSLVerifyFault and -noverify is true.
// In which case, flag.SslThumbprint is set to fault.Thumbprint and the caller should retry the task.
func (flag *HostConnectFlag) AcceptThumbprint(err error) error {
	if f, ok := err.(types.HasFault); ok {
		switch fault := f.Fault().(type) {
		case *types.SSLVerifyFault:
			if flag.noverify {
				flag.SslThumbprint = fault.Thumbprint
				return nil
			}
			return fmt.Errorf("%s Fingerprint is %s", err, fault.Thumbprint)
		}
	}

	return err
}
