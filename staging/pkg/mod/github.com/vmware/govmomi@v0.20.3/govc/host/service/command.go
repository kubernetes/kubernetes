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

package service

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

type service struct {
	*flags.ClientFlag
	*flags.HostSystemFlag
}

func init() {
	cli.Register("host.service", &service{})
}

func (cmd *service) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	cmd.HostSystemFlag, ctx = flags.NewHostSystemFlag(ctx)
	cmd.HostSystemFlag.Register(ctx, f)
}

func (cmd *service) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.HostSystemFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *service) Usage() string {
	return "ACTION ID"
}

func (cmd *service) Description() string {
	return `Apply host service ACTION to service ID.

Where ACTION is one of: start, stop, restart, status, enable, disable

Examples:
  govc host.service enable TSM-SSH
  govc host.service start TSM-SSH`
}

func (cmd *service) status(ctx context.Context, s *object.HostServiceSystem, id string) (string, error) {
	services, err := s.Service(ctx)
	if err != nil {
		return "", err
	}

	for _, service := range services {
		if id == service.Key {
			return Status(service), nil
		}
	}

	return "N/A", nil
}

func (cmd *service) Run(ctx context.Context, f *flag.FlagSet) error {
	host, err := cmd.HostSystem()
	if err != nil {
		return err
	}

	if f.NArg() != 2 {
		return flag.ErrHelp
	}

	arg := f.Arg(0)
	id := f.Arg(1)

	s, err := host.ConfigManager().ServiceSystem(ctx)
	if err != nil {
		return err
	}

	switch arg {
	case "start":
		return s.Start(ctx, id)
	case "stop":
		return s.Stop(ctx, id)
	case "restart":
		return s.Restart(ctx, id)
	case "status":
		ss, err := cmd.status(ctx, s, id)
		if err != nil {
			return err
		}
		fmt.Println(ss)
		return nil
	case "enable":
		return s.UpdatePolicy(ctx, id, string(types.HostServicePolicyOn))
	case "disable":
		return s.UpdatePolicy(ctx, id, string(types.HostServicePolicyOff))
	}

	return flag.ErrHelp
}
