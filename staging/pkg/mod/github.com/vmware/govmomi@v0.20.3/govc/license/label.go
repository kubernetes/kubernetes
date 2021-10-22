/*
Copyright (c) 2019 VMware, Inc. All Rights Reserved.

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

package license

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/license"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type label struct {
	*flags.ClientFlag
}

func init() {
	cli.Register("license.label.set", &label{})
}

func (cmd *label) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
}

func (cmd *label) Usage() string {
	return "LICENSE KEY VAL"
}

func (cmd *label) Description() string {
	return `Set license labels.

Examples:
  govc license.label.set 00000-00000-00000-00000-00000 team cnx # add/set label
  govc license.label.set 00000-00000-00000-00000-00000 team ""  # remove label
  govc license.ls -json | jq '.[] | select(.Labels[].Key == "team") | .LicenseKey'`
}

func (cmd *label) Run(ctx context.Context, f *flag.FlagSet) error {
	client, err := cmd.Client()
	if err != nil {
		return err
	}

	m := license.NewManager(client)

	if f.NArg() != 3 {
		return flag.ErrHelp
	}

	req := types.UpdateLicenseLabel{
		This:       m.Reference(),
		LicenseKey: f.Arg(0),
		LabelKey:   f.Arg(1),
		LabelValue: f.Arg(2),
	}

	_, err = methods.UpdateLicenseLabel(ctx, m.Client(), &req)
	return err
}
