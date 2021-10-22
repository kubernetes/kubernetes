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

package date

import (
	"context"
	"flag"
	"strings"
	"time"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vim25/types"
)

type change struct {
	*flags.HostSystemFlag

	types.HostNtpConfig
	types.HostDateTimeConfig
	date string
}

func init() {
	cli.Register("host.date.change", &change{})
}

type serverConfig types.HostNtpConfig

func (s *serverConfig) String() string {
	return strings.Join(s.Server, ",")
}

func (s *serverConfig) Set(v string) error {
	s.Server = append(s.Server, v)
	return nil
}

func (cmd *change) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)

	f.Var((*serverConfig)(&cmd.HostNtpConfig), "server", "IP or FQDN for NTP server(s)")
	f.StringVar(&cmd.TimeZone, "tz", "", "Change timezone of the host")
	f.StringVar(&cmd.date, "date", "", "Update the date/time on the host")
}

func (cmd *change) Description() string {
	return `Change date and time for HOST.

Examples:
  govc host.date.change -date "$(date -u)"
  govc host.date.change -server time.vmware.com
  govc host.service enable ntpd
  govc host.service start ntpd`
}

func (cmd *change) Process(ctx context.Context) error {
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *change) Run(ctx context.Context, f *flag.FlagSet) error {
	host, err := cmd.HostSystem()
	if err != nil {
		return err
	}

	s, err := host.ConfigManager().DateTimeSystem(ctx)
	if err != nil {
		return err
	}

	if cmd.date != "" {
		d, err := time.Parse(time.UnixDate, cmd.date)
		if err != nil {
			return err
		}
		return s.Update(ctx, d)
	}

	if len(cmd.HostNtpConfig.Server) > 0 {
		cmd.NtpConfig = &cmd.HostNtpConfig
	}

	return s.UpdateConfig(ctx, cmd.HostDateTimeConfig)
}
