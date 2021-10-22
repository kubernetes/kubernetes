/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package association

import (
	"context"
	"flag"
	"fmt"
	"io"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/vapi/rest"
	"github.com/vmware/govmomi/vapi/tags"
)

type ls struct {
	*flags.DatacenterFlag
	r bool
}

func init() {
	cli.Register("tags.attached.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)
	f.BoolVar(&cmd.r, "r", false, "List tags attached to resource")
}

func (cmd *ls) Usage() string {
	return "NAME"
}

func (cmd *ls) Description() string {
	return `List attached tags or objects.

Examples:
  govc tags.attached.ls k8s-region-us
  govc tags.attached.ls -json k8s-zone-us-ca1 | jq .
  govc tags.attached.ls -r /dc1/host/cluster1
  govc tags.attached.ls -json -r /dc1 | jq .`
}

type lsResult []string

func (r lsResult) Write(w io.Writer) error {
	for i := range r {
		fmt.Fprintln(w, r[i])
	}
	return nil
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}
	arg := f.Arg(0)

	return cmd.WithRestClient(ctx, func(c *rest.Client) error {
		var res lsResult
		m := tags.NewManager(c)

		if cmd.r {
			ref, err := convertPath(ctx, cmd.DatacenterFlag, arg)
			if err != nil {
				return err
			}
			attached, err := m.GetAttachedTags(ctx, ref)
			if err != nil {
				return err
			}
			for i := range attached {
				res = append(res, attached[i].Name)
			}
		} else {
			attached, err := m.ListAttachedObjects(ctx, arg)
			if err != nil {
				return err
			}
			for i := range attached {
				res = append(res, attached[i].Reference().String())
			}
		}

		return cmd.WriteResult(res)
	})
}
