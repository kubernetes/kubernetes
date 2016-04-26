/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package ls

import (
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/list"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

type ls struct {
	*flags.DatacenterFlag

	Long bool
	Type string
}

func init() {
	cli.Register("ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	f.BoolVar(&cmd.Long, "l", false, "Long listing format")
	f.StringVar(&cmd.Type, "t", "", "Object type")
}

func (cmd *ls) Process(ctx context.Context) error {
	if err := cmd.DatacenterFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ls) Usage() string {
	return "[PATH]..."
}

func (cmd *ls) typeMatch(ref types.ManagedObjectReference) bool {
	if cmd.Type == "" {
		return true
	}

	return strings.ToLower(cmd.Type) == strings.ToLower(ref.Type)
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	finder, err := cmd.Finder()
	if err != nil {
		return err
	}

	lr := listResult{
		Elements: nil,
		Long:     cmd.Long,
	}

	args := f.Args()
	if len(args) == 0 {
		args = []string{"."}
	}

	for _, arg := range args {
		es, err := finder.ManagedObjectListChildren(context.TODO(), arg)
		if err != nil {
			return err
		}

		for _, e := range es {
			if cmd.typeMatch(e.Object.Reference()) {
				lr.Elements = append(lr.Elements, e)
			}
		}
	}

	return cmd.WriteResult(lr)
}

type listResult struct {
	Elements []list.Element `json:"elements"`

	Long bool `json:"-"`
}

func (l listResult) Write(w io.Writer) error {
	var err error

	for _, e := range l.Elements {
		if !l.Long {
			fmt.Fprintf(w, "%s\n", e.Path)
			continue
		}

		switch e.Object.(type) {
		case mo.Folder:
			if _, err = fmt.Fprintf(w, "%s/\n", e.Path); err != nil {
				return err
			}
		default:
			if _, err = fmt.Fprintf(w, "%s (%s)\n", e.Path, e.Object.Reference().Type); err != nil {
				return err
			}
		}
	}

	return nil
}
