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

package object

import (
	"errors"
	"fmt"
	"path"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

var (
	ErrNotSupported = errors.New("not supported (vCenter only)")
)

// Common contains the fields and functions common to all objects.
type Common struct {
	InventoryPath string

	c *vim25.Client
	r types.ManagedObjectReference
}

func (c Common) String() string {
	ref := fmt.Sprintf("%v", c.Reference())

	if c.InventoryPath == "" {
		return ref
	}

	return fmt.Sprintf("%s @ %s", ref, c.InventoryPath)
}

func NewCommon(c *vim25.Client, r types.ManagedObjectReference) Common {
	return Common{c: c, r: r}
}

func (c Common) Reference() types.ManagedObjectReference {
	return c.r
}

func (c Common) Client() *vim25.Client {
	return c.c
}

// Name returns the base name of the InventoryPath field
func (c Common) Name() string {
	if c.InventoryPath == "" {
		return ""
	}
	return path.Base(c.InventoryPath)
}

func (c *Common) SetInventoryPath(p string) {
	c.InventoryPath = p
}

// ObjectName returns the base name of the InventoryPath field if set,
// otherwise fetches the mo.ManagedEntity.Name field via the property collector.
func (c Common) ObjectName(ctx context.Context) (string, error) {
	var o mo.ManagedEntity

	name := c.Name()
	if name != "" {
		return name, nil
	}

	err := c.Properties(ctx, c.Reference(), []string{"name"}, &o)
	if err != nil {
		return "", err
	}

	return o.Name, nil
}

func (c Common) Properties(ctx context.Context, r types.ManagedObjectReference, ps []string, dst interface{}) error {
	return property.DefaultCollector(c.c).RetrieveOne(ctx, r, ps, dst)
}

func (c Common) Destroy(ctx context.Context) (*Task, error) {
	req := types.Destroy_Task{
		This: c.Reference(),
	}

	res, err := methods.Destroy_Task(ctx, c.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(c.c, res.Returnval), nil
}

func (c Common) Rename(ctx context.Context, name string) (*Task, error) {
	req := types.Rename_Task{
		This:    c.Reference(),
		NewName: name,
	}

	res, err := methods.Rename_Task(ctx, c.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(c.c, res.Returnval), nil
}
