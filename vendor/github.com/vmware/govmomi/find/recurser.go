/*
Copyright (c) 2014-2017 VMware, Inc. All Rights Reserved.

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

package find

import (
	"context"
	"os"
	"path"
	"strings"

	"github.com/vmware/govmomi/list"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25/mo"
)

// spec is used to specify per-search configuration, independent of the Finder instance.
type spec struct {
	// Relative returns the root object to resolve Relative paths (starts with ".")
	Relative func(ctx context.Context) (object.Reference, error)

	// ListMode can be used to optionally force "ls" behavior, rather than "find" behavior
	ListMode *bool

	// Contents configures the Recurser to list the Contents of traversable leaf nodes.
	// This is typically set to true when used from the ls command, where listing
	// a folder means listing its Contents. This is typically set to false for
	// commands that take managed entities that are not folders as input.
	Contents bool

	// Parents specifies the types which can contain the child types being searched for.
	// for example, when searching for a HostSystem, parent types can be
	// "ComputeResource" or "ClusterComputeResource".
	Parents []string

	// Include specifies which types to be included in the results, used only in "find" mode.
	Include []string

	// Nested should be set to types that can be Nested, used only in "find" mode.
	Nested []string

	// ChildType avoids traversing into folders that can't contain the Include types, used only in "find" mode.
	ChildType []string
}

func (s *spec) traversable(o mo.Reference) bool {
	ref := o.Reference()

	switch ref.Type {
	case "Datacenter":
		if len(s.Include) == 1 && s.Include[0] == "Datacenter" {
			// No point in traversing deeper as Datacenters cannot be nested
			return false
		}
		return true
	case "Folder":
		if f, ok := o.(mo.Folder); ok {
			// TODO: Not making use of this yet, but here we can optimize when searching the entire
			// inventory across Datacenters for specific types, for example: 'govc ls -t VirtualMachine /**'
			// should not traverse into a Datacenter's host, network or datatore folders.
			if !s.traversableChildType(f.ChildType) {
				return false
			}
		}

		return true
	}

	for _, kind := range s.Parents {
		if kind == ref.Type {
			return true
		}
	}

	return false
}

func (s *spec) traversableChildType(ctypes []string) bool {
	if len(s.ChildType) == 0 {
		return true
	}

	for _, t := range ctypes {
		for _, c := range s.ChildType {
			if t == c {
				return true
			}
		}
	}

	return false
}

func (s *spec) wanted(e list.Element) bool {
	if len(s.Include) == 0 {
		return true
	}

	w := e.Object.Reference().Type

	for _, kind := range s.Include {
		if w == kind {
			return true
		}
	}

	return false
}

// listMode is a global option to revert to the original Finder behavior,
// disabling the newer "find" mode.
var listMode = os.Getenv("GOVMOMI_FINDER_LIST_MODE") == "true"

func (s *spec) listMode(isPath bool) bool {
	if listMode {
		return true
	}

	if s.ListMode != nil {
		return *s.ListMode
	}

	return isPath
}

type recurser struct {
	Collector *property.Collector

	// All configures the recurses to fetch complete objects for leaf nodes.
	All bool
}

func (r recurser) List(ctx context.Context, s *spec, root list.Element, parts []string) ([]list.Element, error) {
	if len(parts) == 0 {
		// Include non-traversable leaf elements in result. For example, consider
		// the pattern "./vm/my-vm-*", where the pattern should match the VMs and
		// not try to traverse them.
		//
		// Include traversable leaf elements in result, if the contents
		// field is set to false.
		//
		if !s.Contents || !s.traversable(root.Object.Reference()) {
			return []list.Element{root}, nil
		}
	}

	k := list.Lister{
		Collector: r.Collector,
		Reference: root.Object.Reference(),
		Prefix:    root.Path,
	}

	if r.All && len(parts) < 2 {
		k.All = true
	}

	in, err := k.List(ctx)
	if err != nil {
		return nil, err
	}

	// This folder is a leaf as far as the glob goes.
	if len(parts) == 0 {
		return in, nil
	}

	all := parts
	pattern := parts[0]
	parts = parts[1:]

	var out []list.Element
	for _, e := range in {
		matched, err := path.Match(pattern, path.Base(e.Path))
		if err != nil {
			return nil, err
		}

		if !matched {
			matched = strings.HasSuffix(e.Path, "/"+path.Join(all...))
			if matched {
				// name contains a '/'
				out = append(out, e)
			}

			continue
		}

		nres, err := r.List(ctx, s, e, parts)
		if err != nil {
			return nil, err
		}

		out = append(out, nres...)
	}

	return out, nil
}

func (r recurser) Find(ctx context.Context, s *spec, root list.Element, parts []string) ([]list.Element, error) {
	var out []list.Element

	if len(parts) > 0 {
		pattern := parts[0]
		matched, err := path.Match(pattern, path.Base(root.Path))
		if err != nil {
			return nil, err
		}

		if matched && s.wanted(root) {
			out = append(out, root)
		}
	}

	if !s.traversable(root.Object) {
		return out, nil
	}

	k := list.Lister{
		Collector: r.Collector,
		Reference: root.Object.Reference(),
		Prefix:    root.Path,
	}

	in, err := k.List(ctx)
	if err != nil {
		return nil, err
	}

	for _, e := range in {
		nres, err := r.Find(ctx, s, e, parts)
		if err != nil {
			return nil, err
		}

		out = append(out, nres...)
	}

	return out, nil
}
