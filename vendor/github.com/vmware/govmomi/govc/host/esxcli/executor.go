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

package esxcli

import (
	"context"
	"errors"
	"fmt"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vim25/xml"
)

type Executor struct {
	c    *vim25.Client
	host *object.HostSystem
	mme  *types.ReflectManagedMethodExecuter
	dtm  *types.InternalDynamicTypeManager
	info map[string]*CommandInfo
}

func NewExecutor(c *vim25.Client, host *object.HostSystem) (*Executor, error) {
	ctx := context.TODO()
	e := &Executor{
		c:    c,
		host: host,
		info: make(map[string]*CommandInfo),
	}

	{
		req := types.RetrieveManagedMethodExecuter{
			This: host.Reference(),
		}

		res, err := methods.RetrieveManagedMethodExecuter(ctx, c, &req)
		if err != nil {
			return nil, err
		}

		e.mme = res.Returnval
	}

	{
		req := types.RetrieveDynamicTypeManager{
			This: host.Reference(),
		}

		res, err := methods.RetrieveDynamicTypeManager(ctx, c, &req)
		if err != nil {
			return nil, err
		}

		e.dtm = res.Returnval
	}

	return e, nil
}

func (e *Executor) CommandInfo(c *Command) (*CommandInfoMethod, error) {
	ns := c.Namespace()
	var info *CommandInfo
	var ok bool

	if info, ok = e.info[ns]; !ok {
		req := types.ExecuteSoap{
			Moid:   "ha-dynamic-type-manager-local-cli-cliinfo",
			Method: "vim.CLIInfo.FetchCLIInfo",
			Argument: []types.ReflectManagedMethodExecuterSoapArgument{
				c.Argument("typeName", "vim.EsxCLI."+ns),
			},
		}

		info = new(CommandInfo)
		if err := e.Execute(&req, info); err != nil {
			return nil, err
		}

		e.info[ns] = info
	}

	name := c.Name()
	for _, method := range info.Method {
		if method.Name == name {
			return method, nil
		}
	}

	return nil, fmt.Errorf("method '%s' not found in name space '%s'", name, c.Namespace())
}

func (e *Executor) NewRequest(args []string) (*types.ExecuteSoap, *CommandInfoMethod, error) {
	c := NewCommand(args)

	info, err := e.CommandInfo(c)
	if err != nil {
		return nil, nil, err
	}

	sargs, err := c.Parse(info.Param)
	if err != nil {
		return nil, nil, err
	}

	sreq := types.ExecuteSoap{
		Moid:     c.Moid(),
		Method:   c.Method(),
		Argument: sargs,
	}

	return &sreq, info, nil
}

func (e *Executor) Execute(req *types.ExecuteSoap, res interface{}) error {
	ctx := context.TODO()
	req.This = e.mme.ManagedObjectReference
	req.Version = "urn:vim25/5.0"

	x, err := methods.ExecuteSoap(ctx, e.c, req)
	if err != nil {
		return err
	}

	if x.Returnval != nil {
		if x.Returnval.Fault != nil {
			return errors.New(x.Returnval.Fault.FaultMsg)
		}

		if err := xml.Unmarshal([]byte(x.Returnval.Response), res); err != nil {
			return err
		}
	}

	return nil
}

func (e *Executor) Run(args []string) (*Response, error) {
	req, info, err := e.NewRequest(args)
	if err != nil {
		return nil, err
	}

	res := &Response{
		Info: info,
	}

	if err := e.Execute(req, res); err != nil {
		return nil, err
	}

	return res, nil
}
