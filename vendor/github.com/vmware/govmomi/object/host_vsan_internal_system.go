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

package object

import (
	"context"
	"encoding/json"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

type HostVsanInternalSystem struct {
	Common
}

func NewHostVsanInternalSystem(c *vim25.Client, ref types.ManagedObjectReference) *HostVsanInternalSystem {
	m := HostVsanInternalSystem{
		Common: NewCommon(c, ref),
	}

	return &m
}

// QueryVsanObjectUuidsByFilter returns vSAN DOM object uuids by filter.
func (m HostVsanInternalSystem) QueryVsanObjectUuidsByFilter(ctx context.Context, uuids []string, limit int32, version int32) ([]string, error) {
	req := types.QueryVsanObjectUuidsByFilter{
		This:    m.Reference(),
		Uuids:   uuids,
		Limit:   limit,
		Version: version,
	}

	res, err := methods.QueryVsanObjectUuidsByFilter(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

type VsanObjExtAttrs struct {
	Type  string `json:"Object type"`
	Class string `json:"Object class"`
	Size  string `json:"Object size"`
	Path  string `json:"Object path"`
	Name  string `json:"User friendly name"`
}

func (a *VsanObjExtAttrs) DatastorePath(dir string) string {
	l := len(dir)
	path := a.Path

	if len(path) >= l {
		path = a.Path[l:]
	}

	if path != "" {
		return path
	}

	return a.Name // vmnamespace
}

// GetVsanObjExtAttrs is internal and intended for troubleshooting/debugging situations in the field.
// WARNING: This API can be slow because we do IOs (reads) to all the objects.
func (m HostVsanInternalSystem) GetVsanObjExtAttrs(ctx context.Context, uuids []string) (map[string]VsanObjExtAttrs, error) {
	req := types.GetVsanObjExtAttrs{
		This:  m.Reference(),
		Uuids: uuids,
	}

	res, err := methods.GetVsanObjExtAttrs(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	var attrs map[string]VsanObjExtAttrs

	err = json.Unmarshal([]byte(res.Returnval), &attrs)

	return attrs, err
}

// DeleteVsanObjects is internal and intended for troubleshooting/debugging only.
// WARNING: This API can be slow because we do IOs to all the objects.
// DOM won't allow access to objects which have lost quorum. Such objects can be deleted with the optional "force" flag.
// These objects may however re-appear with quorum if the absent components come back (network partition gets resolved, etc.)
func (m HostVsanInternalSystem) DeleteVsanObjects(ctx context.Context, uuids []string, force *bool) ([]types.HostVsanInternalSystemDeleteVsanObjectsResult, error) {
	req := types.DeleteVsanObjects{
		This:  m.Reference(),
		Uuids: uuids,
		Force: force,
	}

	res, err := methods.DeleteVsanObjects(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}
