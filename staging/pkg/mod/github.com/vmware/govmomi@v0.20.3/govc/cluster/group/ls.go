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

package group

import (
	"context"
	"flag"
	"fmt"
	"io"

	"github.com/vmware/govmomi/govc/cli"
)

type ls struct {
	*InfoFlag
}

func init() {
	cli.Register("cluster.group.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.InfoFlag, ctx = NewInfoFlag(ctx)
	cmd.InfoFlag.Register(ctx, f)
}

func (cmd *ls) Process(ctx context.Context) error {
	return cmd.InfoFlag.Process(ctx)
}

func (cmd *ls) Description() string {
	return `List cluster groups and group members.

Examples:
  govc cluster.group.ls -cluster my_cluster
  govc cluster.group.ls -cluster my_cluster -name my_group`
}

type groupResult []string

func (r groupResult) Write(w io.Writer) error {
	for i := range r {
		fmt.Fprintln(w, r[i])
	}

	return nil
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	var res groupResult

	if cmd.name == "" {
		groups, err := cmd.Groups(ctx)
		if err != nil {
			return err
		}

		for _, g := range groups {
			res = append(res, g.GetClusterGroupInfo().Name)
		}
	} else {
		group, err := cmd.Group(ctx)
		if err != nil {
			return err
		}

		names, err := cmd.Names(ctx, *group.refs)
		if err != nil {
			return err
		}

		for _, ref := range *group.refs {
			res = append(res, names[ref])
		}
	}

	return cmd.WriteResult(res)
}
