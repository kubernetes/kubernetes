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

package fields

import (
	"context"
	"flag"
	"fmt"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
)

type add struct {
	*flags.ClientFlag
	kind string
}

func init() {
	cli.Register("fields.add", &add{})
}

func (cmd *add) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	f.StringVar(&cmd.kind, "type", "", "Managed object type")
}

func (cmd *add) Usage() string {
	return "NAME"
}

func (cmd *add) Description() string {
	return `Add a custom field type with NAME.

Examples:
  govc fields.add my-field-name # adds a field to all managed object types
  govc fields.add -type VirtualMachine my-vm-field-name # adds a field to the VirtualMachine type`
}

func (cmd *add) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}

	c, err := cmd.Client()
	if err != nil {
		return err
	}

	m, err := object.GetCustomFieldsManager(c)
	if err != nil {
		return err
	}

	name := f.Arg(0)

	def, err := m.Add(ctx, name, cmd.kind, nil, nil)
	if err != nil {
		return err
	}

	fmt.Printf("%d\n", def.Key)

	return nil
}
