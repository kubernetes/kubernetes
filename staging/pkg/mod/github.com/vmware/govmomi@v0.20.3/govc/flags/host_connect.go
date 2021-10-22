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
	"context"
	"flag"
	"fmt"
	"net/http"
	"net/url"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25"
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
		f.StringVar(&flag.SslThumbprint, "thumbprint", "", "SHA-1 thumbprint of the host's SSL certificate")
		f.BoolVar(&flag.Force, "force", false, "Force when host is managed by another VC")

		f.BoolVar(&flag.noverify, "noverify", false, "Accept host thumbprint without verification")
	})
}

func (flag *HostConnectFlag) Process(ctx context.Context) error {
	return nil
}

// Spec attempts to fill in SslThumbprint if empty.
// First checks GOVC_TLS_KNOWN_HOSTS, if not found and noverify=true then
// use object.HostCertificateInfo to get the thumbprint.
func (flag *HostConnectFlag) Spec(c *vim25.Client) types.HostConnectSpec {
	spec := flag.HostConnectSpec

	if spec.SslThumbprint == "" {
		spec.SslThumbprint = c.Thumbprint(spec.HostName)

		if spec.SslThumbprint == "" && flag.noverify {
			var info object.HostCertificateInfo
			t := c.Transport.(*http.Transport)
			_ = info.FromURL(&url.URL{Host: spec.HostName}, t.TLSClientConfig)
			spec.SslThumbprint = info.ThumbprintSHA1
		}
	}

	return spec
}

// Fault checks if error is SSLVerifyFault, including the thumbprint if so
func (flag *HostConnectFlag) Fault(err error) error {
	if err == nil {
		return nil
	}

	if f, ok := err.(types.HasFault); ok {
		switch fault := f.Fault().(type) {
		case *types.SSLVerifyFault:
			return fmt.Errorf("%s thumbprint=%s", err, fault.Thumbprint)
		}
	}

	return err
}
