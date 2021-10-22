/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package option

import (
	"context"
	"flag"
	"fmt"
	"strconv"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type Set struct {
	*flags.ClientFlag
}

func init() {
	cli.Register("option.set", &Set{})
}

func (cmd *Set) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
}

func (cmd *Set) Process(ctx context.Context) error {
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *Set) Usage() string {
	return "NAME VALUE"
}

var SetDescription = `Set option NAME to VALUE.`

func (cmd *Set) Description() string {
	return SetDescription + `

Examples:
  govc option.set log.level info
  govc option.set logger.Vsan verbose`
}

func isInvalidName(err error) bool {
	if soap.IsSoapFault(err) {
		soapFault := soap.ToSoapFault(err)
		if _, ok := soapFault.VimFault().(types.InvalidName); ok {
			return ok
		}
	}

	return false
}

func (cmd *Set) Update(ctx context.Context, f *flag.FlagSet, m *object.OptionManager) error {
	if f.NArg() != 2 {
		return flag.ErrHelp
	}

	name := f.Arg(0)
	val := f.Arg(1)

	opts, err := m.Query(ctx, name)
	if err != nil {
		if isInvalidName(err) {
			// If the option doesn't exist, creating one can only have a string Value.
			// The Key prefix is limited in this case too, it seems to the config.* namespace.
			return m.Update(ctx, []types.BaseOptionValue{&types.OptionValue{
				Key:   name,
				Value: val,
			}})
		}
		return err
	}

	if len(opts) != 1 {
		return flag.ErrHelp
	}

	var set types.AnyType

	switch x := opts[0].GetOptionValue().Value.(type) {
	case string:
		set = val
	case bool:
		set, err = strconv.ParseBool(val)
		if err != nil {
			return err
		}
	case int32:
		s, err := strconv.ParseInt(val, 10, 32)
		if err != nil {
			return err
		}
		set = s
	case int64:
		set, err = strconv.ParseInt(val, 10, 64)
		if err != nil {
			return err
		}
	default:
		return fmt.Errorf("type %T conversion not supported", x)
	}

	opts[0].GetOptionValue().Value = set

	return m.Update(ctx, opts)
}

func (cmd *Set) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	m := object.NewOptionManager(c, *c.ServiceContent.Setting)

	return cmd.Update(ctx, f, m)
}
