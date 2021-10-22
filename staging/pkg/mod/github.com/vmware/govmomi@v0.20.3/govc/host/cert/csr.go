/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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

package cert

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type csr struct {
	*flags.HostSystemFlag

	ip bool
}

func init() {
	cli.Register("host.cert.csr", &csr{})
}

func (cmd *csr) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	f.BoolVar(&cmd.ip, "ip", false, "Use IP address as CN")
}

func (cmd *csr) Description() string {
	return `Generate a certificate-signing request (CSR) for HOST.`
}

func (cmd *csr) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *csr) Run(ctx context.Context, f *flag.FlagSet) error {
	host, err := cmd.HostSystem()
	if err != nil {
		return err
	}

	m, err := host.ConfigManager().CertificateManager(ctx)
	if err != nil {
		return err
	}

	output, err := m.GenerateCertificateSigningRequest(ctx, cmd.ip)
	if err != nil {
		return err
	}

	_, err = fmt.Println(output)
	return err
}
