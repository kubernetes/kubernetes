/*
Copyright (c) 2014-2016 VMware, Inc. All Rights Reserved.

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

package datastore

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path"
	"strings"
	"text/tabwriter"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/units"
	"github.com/vmware/govmomi/vim25/types"
)

type ls struct {
	*flags.DatastoreFlag
	*flags.OutputFlag

	long    bool
	slash   bool
	all     bool
	recurse bool
}

func init() {
	cli.Register("datastore.ls", &ls{})
}

func (cmd *ls) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.DatastoreFlag, ctx = flags.NewDatastoreFlag(ctx)
	cmd.DatastoreFlag.Register(ctx, f)

	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	f.BoolVar(&cmd.long, "l", false, "Long listing format")
	f.BoolVar(&cmd.slash, "p", false, "Append / indicator to directories")
	f.BoolVar(&cmd.all, "a", false, "Do not ignore entries starting with .")
	f.BoolVar(&cmd.recurse, "R", false, "List subdirectories recursively")
}

func (cmd *ls) Process(ctx context.Context) error {
	if err := cmd.DatastoreFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ls) Usage() string {
	return "[FILE]..."
}

func isInvalid(err error) bool {
	if f, ok := err.(types.HasFault); ok {
		switch f.Fault().(type) {
		case *types.InvalidArgument:
			return true
		}
	}

	return false
}

func (cmd *ls) Run(ctx context.Context, f *flag.FlagSet) error {
	args := cmd.Args(f.Args())

	ds, err := cmd.Datastore()
	if err != nil {
		return err
	}

	b, err := ds.Browser(ctx)
	if err != nil {
		return err
	}

	if len(args) == 0 {
		args = append(args, object.DatastorePath{})
	}

	result := &listOutput{
		rs:  make([]types.HostDatastoreBrowserSearchResults, 0),
		cmd: cmd,
	}

	for _, p := range args {
		arg := p.Path

		spec := types.HostDatastoreBrowserSearchSpec{
			MatchPattern: []string{"*"},
		}

		if cmd.long {
			spec.Details = &types.FileQueryFlags{
				FileType:     true,
				FileSize:     true,
				FileOwner:    types.NewBool(true), // TODO: omitempty is generated, but seems to be required
				Modification: true,
			}
		}

		for i := 0; ; i++ {
			r, err := cmd.ListPath(b, arg, spec)
			if err != nil {
				// Treat the argument as a match pattern if not found as directory
				if i == 0 && types.IsFileNotFound(err) || isInvalid(err) {
					spec.MatchPattern[0] = path.Base(arg)
					arg = path.Dir(arg)
					continue
				}

				return err
			}

			// Treat an empty result against match pattern as file not found
			if i == 1 && len(r) == 1 && len(r[0].File) == 0 {
				return fmt.Errorf("File %s/%s was not found", r[0].FolderPath, spec.MatchPattern[0])
			}

			for n := range r {
				result.add(r[n])
			}

			break
		}
	}

	return cmd.WriteResult(result)
}

func (cmd *ls) ListPath(b *object.HostDatastoreBrowser, path string, spec types.HostDatastoreBrowserSearchSpec) ([]types.HostDatastoreBrowserSearchResults, error) {
	ctx := context.TODO()

	path, err := cmd.DatastorePath(path)
	if err != nil {
		return nil, err
	}

	search := b.SearchDatastore
	if cmd.recurse {
		search = b.SearchDatastoreSubFolders
	}

	task, err := search(ctx, path, &spec)
	if err != nil {
		return nil, err
	}

	info, err := task.WaitForResult(ctx, nil)
	if err != nil {
		return nil, err
	}

	switch r := info.Result.(type) {
	case types.HostDatastoreBrowserSearchResults:
		return []types.HostDatastoreBrowserSearchResults{r}, nil
	case types.ArrayOfHostDatastoreBrowserSearchResults:
		return r.HostDatastoreBrowserSearchResults, nil
	default:
		panic(fmt.Sprintf("unknown result type: %T", r))
	}
}

type listOutput struct {
	rs  []types.HostDatastoreBrowserSearchResults
	cmd *ls
}

func (o *listOutput) add(r types.HostDatastoreBrowserSearchResults) {
	if o.cmd.recurse && !o.cmd.all {
		// filter out ".hidden" directories
		path := strings.SplitN(r.FolderPath, " ", 2)
		if len(path) == 2 {
			path = strings.Split(path[1], "/")
			if path[0] == "." {
				path = path[1:]
			}

			for _, p := range path {
				if len(p) != 0 && p[0] == '.' {
					return
				}
			}
		}
	}

	res := r
	res.File = nil

	for _, f := range r.File {
		if f.GetFileInfo().Path[0] == '.' && !o.cmd.all {
			continue
		}

		if o.cmd.slash {
			if d, ok := f.(*types.FolderFileInfo); ok {
				d.Path += "/"
			}
		}

		res.File = append(res.File, f)
	}

	o.rs = append(o.rs, res)
}

// hasMultiplePaths returns whether or not the slice of search results contains
// results from more than one folder path.
func (o *listOutput) hasMultiplePaths() bool {
	if len(o.rs) == 0 {
		return false
	}

	p := o.rs[0].FolderPath

	// Multiple paths if any entry is not equal to the first one.
	for _, e := range o.rs {
		if e.FolderPath != p {
			return true
		}
	}

	return false
}

func (o *listOutput) MarshalJSON() ([]byte, error) {
	return json.Marshal(o.rs)
}

func (o *listOutput) Write(w io.Writer) error {
	// Only include path header if we're dealing with more than one path.
	includeHeader := false
	if o.hasMultiplePaths() {
		includeHeader = true
	}

	tw := tabwriter.NewWriter(w, 3, 0, 2, ' ', 0)
	for i, r := range o.rs {
		if includeHeader {
			if i > 0 {
				fmt.Fprintf(tw, "\n")
			}
			fmt.Fprintf(tw, "%s:\n", r.FolderPath)
		}
		for _, file := range r.File {
			info := file.GetFileInfo()
			if o.cmd.long {
				fmt.Fprintf(tw, "%s\t%s\t%s\n", units.ByteSize(info.FileSize), info.Modification.Format("Mon Jan 2 15:04:05 2006"), info.Path)
			} else {
				fmt.Fprintf(tw, "%s\n", info.Path)
			}
		}
	}
	tw.Flush()
	return nil
}
