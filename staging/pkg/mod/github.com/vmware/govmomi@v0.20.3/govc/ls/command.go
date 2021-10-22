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
	"context"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/list"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

type ls struct {
	*flags.DatacenterFlag

	Long  bool
	Type  string
	ToRef bool
	DeRef bool
}

func init() {
	cli.Register("ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatacenterFlag, ctx = flags.NewDatacenterFlag(ctx)
	cmd.DatacenterFlag.Register(ctx, f)

	f.BoolVar(&cmd.Long, "l", false, "Long listing format")
	f.BoolVar(&cmd.ToRef, "i", false, "Print the managed object reference")
	f.BoolVar(&cmd.DeRef, "L", false, "Follow managed object references")
	f.StringVar(&cmd.Type, "t", "", "Object type")
}

func (cmd *ls) Description() string {
	return `List inventory items.

Examples:
  govc ls -l '*'
  govc ls -t ClusterComputeResource host
  govc ls -t Datastore host/ClusterA/* | grep -v local | xargs -n1 basename | sort | uniq`
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
	finder, err := cmd.Finder(cmd.All())
	if err != nil {
		return err
	}

	lr := listResult{
		ls:       cmd,
		Elements: nil,
	}

	args := f.Args()
	if len(args) == 0 {
		args = []string{"."}
	}

	var ref = new(types.ManagedObjectReference)

	var types []string
	if cmd.Type != "" {
		// TODO: support multiple -t flags
		types = []string{cmd.Type}
	}

	for _, arg := range args {
		if cmd.DeRef && ref.FromString(arg) {
			e, err := finder.Element(ctx, *ref)
			if err == nil {
				if cmd.typeMatch(*ref) {
					if e.Path == "/" && ref.Type != "Folder" {
						// Special case: when given a moref with no ancestors,
						// just echo the moref.
						e.Path = ref.String()
					}
					lr.Elements = append(lr.Elements, *e)
				}
				continue
			}
		}

		es, err := finder.ManagedObjectListChildren(ctx, arg, types...)
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
	*ls      `json:"-"`
	Elements []list.Element `json:"elements"`
}

func (l listResult) Write(w io.Writer) error {
	var err error

	for _, e := range l.Elements {
		if l.ToRef {
			fmt.Fprint(w, e.Object.Reference().String())
			if l.Long {
				fmt.Fprintf(w, " %s", e.Path)
			}
			fmt.Fprintln(w)
			continue
		}

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
