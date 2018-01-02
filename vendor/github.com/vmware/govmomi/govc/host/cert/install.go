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
	"bytes"
	"context"
	"flag"
	"io"
	"io/ioutil"
	"os"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type install struct {
	*flags.HostSystemFlag
}

func init() {
	cli.Register("host.cert.import", &install{})
}

func (cmd *install) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)
}

func (cmd *install) Usage() string {
	return "FILE"
}

func (cmd *install) Description() string {
	return `Install SSL certificate FILE on HOST.

If FILE name is "-", read certificate from stdin.`
}

func (cmd *install) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *install) Run(ctx context.Context, f *flag.FlagSet) error {
	host, err := cmd.HostSystem()
	if err != nil {
		return err
	}

	m, err := host.ConfigManager().CertificateManager(ctx)
	if err != nil {
		return err
	}

	var cert string

	name := f.Arg(0)
	if name == "-" || name == "" {
		var buf bytes.Buffer
		if _, err := io.Copy(&buf, os.Stdin); err != nil {
			return err
		}
		cert = buf.String()
	} else {
		b, err := ioutil.ReadFile(name)
		if err != nil {
			return err
		}
		cert = string(b)
	}

	return m.InstallServerCertificate(ctx, cert)
}
