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
	"context"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type DiagnosticManager struct {
	Common
}

func NewDiagnosticManager(c *vim25.Client) *DiagnosticManager {
	m := DiagnosticManager{
		Common: NewCommon(c, *c.ServiceContent.DiagnosticManager),
	}

	return &m
}

func (m DiagnosticManager) Log(ctx context.Context, host *HostSystem, key string) *DiagnosticLog {
	return &DiagnosticLog{
		m:    m,
		Key:  key,
		Host: host,
	}
}

func (m DiagnosticManager) BrowseLog(ctx context.Context, host *HostSystem, key string, start, lines int32) (*types.DiagnosticManagerLogHeader, error) {
	req := types.BrowseDiagnosticLog{
		This:  m.Reference(),
		Key:   key,
		Start: start,
		Lines: lines,
	}

	if host != nil {
		ref := host.Reference()
		req.Host = &ref
	}

	res, err := methods.BrowseDiagnosticLog(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

func (m DiagnosticManager) GenerateLogBundles(ctx context.Context, includeDefault bool, host []*HostSystem) (*Task, error) {
	req := types.GenerateLogBundles_Task{
		This:           m.Reference(),
		IncludeDefault: includeDefault,
	}

	for _, h := range host {
		req.Host = append(req.Host, h.Reference())
	}

	res, err := methods.GenerateLogBundles_Task(ctx, m.c, &req)
	if err != nil {
		return nil, err
	}

	return NewTask(m.c, res.Returnval), nil
}

func (m DiagnosticManager) QueryDescriptions(ctx context.Context, host *HostSystem) ([]types.DiagnosticManagerLogDescriptor, error) {
	req := types.QueryDescriptions{
		This: m.Reference(),
	}

	if host != nil {
		ref := host.Reference()
		req.Host = &ref
	}

	res, err := methods.QueryDescriptions(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}
