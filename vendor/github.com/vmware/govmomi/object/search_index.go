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

type SearchIndex struct {
	Common
}

func NewSearchIndex(c *vim25.Client) *SearchIndex {
	s := SearchIndex{
		Common: NewCommon(c, *c.ServiceContent.SearchIndex),
	}

	return &s
}

// FindByDatastorePath finds a virtual machine by its location on a datastore.
func (s SearchIndex) FindByDatastorePath(ctx context.Context, dc *Datacenter, path string) (Reference, error) {
	req := types.FindByDatastorePath{
		This:       s.Reference(),
		Datacenter: dc.Reference(),
		Path:       path,
	}

	res, err := methods.FindByDatastorePath(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	if res.Returnval == nil {
		return nil, nil
	}
	return NewReference(s.c, *res.Returnval), nil
}

// FindByDnsName finds a virtual machine or host by DNS name.
func (s SearchIndex) FindByDnsName(ctx context.Context, dc *Datacenter, dnsName string, vmSearch bool) (Reference, error) {
	req := types.FindByDnsName{
		This:     s.Reference(),
		DnsName:  dnsName,
		VmSearch: vmSearch,
	}
	if dc != nil {
		ref := dc.Reference()
		req.Datacenter = &ref
	}

	res, err := methods.FindByDnsName(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	if res.Returnval == nil {
		return nil, nil
	}
	return NewReference(s.c, *res.Returnval), nil
}

// FindByInventoryPath finds a managed entity based on its location in the inventory.
func (s SearchIndex) FindByInventoryPath(ctx context.Context, path string) (Reference, error) {
	req := types.FindByInventoryPath{
		This:          s.Reference(),
		InventoryPath: path,
	}

	res, err := methods.FindByInventoryPath(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	if res.Returnval == nil {
		return nil, nil
	}

	r := NewReference(s.c, *res.Returnval)

	type common interface {
		SetInventoryPath(string)
	}

	if c, ok := r.(common); ok {
		c.SetInventoryPath(path)
	}

	return r, nil
}

// FindByIp finds a virtual machine or host by IP address.
func (s SearchIndex) FindByIp(ctx context.Context, dc *Datacenter, ip string, vmSearch bool) (Reference, error) {
	req := types.FindByIp{
		This:     s.Reference(),
		Ip:       ip,
		VmSearch: vmSearch,
	}
	if dc != nil {
		ref := dc.Reference()
		req.Datacenter = &ref
	}

	res, err := methods.FindByIp(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	if res.Returnval == nil {
		return nil, nil
	}
	return NewReference(s.c, *res.Returnval), nil
}

// FindByUuid finds a virtual machine or host by UUID.
func (s SearchIndex) FindByUuid(ctx context.Context, dc *Datacenter, uuid string, vmSearch bool, instanceUuid *bool) (Reference, error) {
	req := types.FindByUuid{
		This:         s.Reference(),
		Uuid:         uuid,
		VmSearch:     vmSearch,
		InstanceUuid: instanceUuid,
	}
	if dc != nil {
		ref := dc.Reference()
		req.Datacenter = &ref
	}

	res, err := methods.FindByUuid(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	if res.Returnval == nil {
		return nil, nil
	}
	return NewReference(s.c, *res.Returnval), nil
}

// FindChild finds a particular child based on a managed entity name.
func (s SearchIndex) FindChild(ctx context.Context, entity Reference, name string) (Reference, error) {
	req := types.FindChild{
		This:   s.Reference(),
		Entity: entity.Reference(),
		Name:   name,
	}

	res, err := methods.FindChild(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	if res.Returnval == nil {
		return nil, nil
	}
	return NewReference(s.c, *res.Returnval), nil
}

// FindAllByDnsName finds all virtual machines or hosts by DNS name.
func (s SearchIndex) FindAllByDnsName(ctx context.Context, dc *Datacenter, dnsName string, vmSearch bool) ([]Reference, error) {
	req := types.FindAllByDnsName{
		This:     s.Reference(),
		DnsName:  dnsName,
		VmSearch: vmSearch,
	}
	if dc != nil {
		ref := dc.Reference()
		req.Datacenter = &ref
	}

	res, err := methods.FindAllByDnsName(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	if len(res.Returnval) == 0 {
		return nil, nil
	}

	var references []Reference
	for _, returnval := range res.Returnval {
		references = append(references, NewReference(s.c, returnval))
	}
	return references, nil
}

// FindAllByIp finds all virtual machines or hosts by IP address.
func (s SearchIndex) FindAllByIp(ctx context.Context, dc *Datacenter, ip string, vmSearch bool) ([]Reference, error) {
	req := types.FindAllByIp{
		This:     s.Reference(),
		Ip:       ip,
		VmSearch: vmSearch,
	}
	if dc != nil {
		ref := dc.Reference()
		req.Datacenter = &ref
	}

	res, err := methods.FindAllByIp(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	if len(res.Returnval) == 0 {
		return nil, nil
	}

	var references []Reference
	for _, returnval := range res.Returnval {
		references = append(references, NewReference(s.c, returnval))
	}
	return references, nil
}

// FindAllByUuid finds all virtual machines or hosts by UUID.
func (s SearchIndex) FindAllByUuid(ctx context.Context, dc *Datacenter, uuid string, vmSearch bool, instanceUuid *bool) ([]Reference, error) {
	req := types.FindAllByUuid{
		This:         s.Reference(),
		Uuid:         uuid,
		VmSearch:     vmSearch,
		InstanceUuid: instanceUuid,
	}
	if dc != nil {
		ref := dc.Reference()
		req.Datacenter = &ref
	}

	res, err := methods.FindAllByUuid(ctx, s.c, &req)
	if err != nil {
		return nil, err
	}

	if len(res.Returnval) == 0 {
		return nil, nil
	}

	var references []Reference
	for _, returnval := range res.Returnval {
		references = append(references, NewReference(s.c, returnval))
	}
	return references, nil
}
