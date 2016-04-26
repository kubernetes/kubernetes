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

package list

import (
	"path"
	"path/filepath"

	"github.com/vmware/govmomi/property"
	"golang.org/x/net/context"
)

type Recurser struct {
	Collector *property.Collector

	// All configures the recurses to fetch complete objects for leaf nodes.
	All bool

	// TraverseLeafs configures the Recurser to traverse traversable leaf nodes.
	// This is typically set to true when used from the ls command, where listing
	// a folder means listing its contents. This is typically set to false for
	// commands that take managed entities that are not folders as input.
	TraverseLeafs bool
}

func (r Recurser) Recurse(ctx context.Context, root Element, parts []string) ([]Element, error) {
	if len(parts) == 0 {
		// Include non-traversable leaf elements in result. For example, consider
		// the pattern "./vm/my-vm-*", where the pattern should match the VMs and
		// not try to traverse them.
		//
		// Include traversable leaf elements in result, if the TraverseLeafs
		// field is set to false.
		//
		if !traversable(root.Object.Reference()) || !r.TraverseLeafs {
			return []Element{root}, nil
		}
	}

	k := Lister{
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

	pattern := parts[0]
	parts = parts[1:]

	var out []Element
	for _, e := range in {
		matched, err := filepath.Match(pattern, path.Base(e.Path))
		if err != nil {
			return nil, err
		}

		if !matched {
			continue
		}

		nres, err := r.Recurse(ctx, e, parts)
		if err != nil {
			return nil, err
		}

		out = append(out, nres...)
	}

	return out, nil
}
