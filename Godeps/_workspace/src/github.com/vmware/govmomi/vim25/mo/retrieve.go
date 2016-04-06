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

package mo

import (
	"reflect"

	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

func ignoreMissingProperty(ref types.ManagedObjectReference, p types.MissingProperty) bool {
	switch ref.Type {
	case "VirtualMachine":
		switch p.Path {
		case "environmentBrowser":
			// See https://github.com/vmware/govmomi/pull/242
			return true
		case "alarmActionsEnabled":
			// Seen with vApp child VM
			return true
		}
	}

	return false
}

// ObjectContentToType loads an ObjectContent value into the value it
// represents. If the ObjectContent value has a non-empty 'MissingSet' field,
// it returns the first fault it finds there as error. If the 'MissingSet'
// field is empty, it returns a pointer to a reflect.Value. It handles contain
// nested properties, such as 'guest.ipAddress' or 'config.hardware'.
func ObjectContentToType(o types.ObjectContent) (interface{}, error) {
	// Expect no properties in the missing set
	for _, p := range o.MissingSet {
		if ignoreMissingProperty(o.Obj, p) {
			continue
		}

		return nil, soap.WrapVimFault(p.Fault.Fault)
	}

	ti := typeInfoForType(o.Obj.Type)
	v, err := ti.LoadFromObjectContent(o)
	if err != nil {
		return nil, err
	}

	return v.Elem().Interface(), nil
}

// LoadRetrievePropertiesResponse converts the response of a call to
// RetrieveProperties to one or more managed objects.
func LoadRetrievePropertiesResponse(res *types.RetrievePropertiesResponse, dst interface{}) error {
	rt := reflect.TypeOf(dst)
	if rt == nil || rt.Kind() != reflect.Ptr {
		panic("need pointer")
	}

	rv := reflect.ValueOf(dst).Elem()
	if !rv.CanSet() {
		panic("cannot set dst")
	}

	isSlice := false
	switch rt.Elem().Kind() {
	case reflect.Struct:
	case reflect.Slice:
		isSlice = true
	default:
		panic("unexpected type")
	}

	if isSlice {
		for _, p := range res.Returnval {
			v, err := ObjectContentToType(p)
			if err != nil {
				return err
			}

			vt := reflect.TypeOf(v)

			if !rv.Type().AssignableTo(vt) {
				// For example: dst is []ManagedEntity, res is []HostSystem
				if field, ok := vt.FieldByName(rt.Elem().Elem().Name()); ok && field.Anonymous {
					rv.Set(reflect.Append(rv, reflect.ValueOf(v).FieldByIndex(field.Index)))
					continue
				}
			}

			rv.Set(reflect.Append(rv, reflect.ValueOf(v)))
		}
	} else {
		switch len(res.Returnval) {
		case 0:
		case 1:
			v, err := ObjectContentToType(res.Returnval[0])
			if err != nil {
				return err
			}

			vt := reflect.TypeOf(v)

			if !rv.Type().AssignableTo(vt) {
				// For example: dst is ComputeResource, res is ClusterComputeResource
				if field, ok := vt.FieldByName(rt.Elem().Name()); ok && field.Anonymous {
					rv.Set(reflect.ValueOf(v).FieldByIndex(field.Index))
					return nil
				}
			}

			rv.Set(reflect.ValueOf(v))
		default:
			// If dst is not a slice, expect to receive 0 or 1 results
			panic("more than 1 result")
		}
	}

	return nil
}

// RetrievePropertiesForRequest calls the RetrieveProperties method with the
// specified request and decodes the response struct into the value pointed to
// by dst.
func RetrievePropertiesForRequest(ctx context.Context, r soap.RoundTripper, req types.RetrieveProperties, dst interface{}) error {
	res, err := methods.RetrieveProperties(ctx, r, &req)
	if err != nil {
		return err
	}

	return LoadRetrievePropertiesResponse(res, dst)
}

// RetrieveProperties retrieves the properties of the managed object specified
// as obj and decodes the response struct into the value pointed to by dst.
func RetrieveProperties(ctx context.Context, r soap.RoundTripper, pc, obj types.ManagedObjectReference, dst interface{}) error {
	req := types.RetrieveProperties{
		This: pc,
		SpecSet: []types.PropertyFilterSpec{
			{
				ObjectSet: []types.ObjectSpec{
					{
						Obj:  obj,
						Skip: types.NewBool(false),
					},
				},
				PropSet: []types.PropertySpec{
					{
						All:  types.NewBool(true),
						Type: obj.Type,
					},
				},
			},
		},
	}

	return RetrievePropertiesForRequest(ctx, r, req, dst)
}
