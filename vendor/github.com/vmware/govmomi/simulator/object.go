/*
Copyright (c) 2017-2018 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"bytes"

	"github.com/google/uuid"

	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vim25/xml"
)

func SetCustomValue(ctx *Context, req *types.SetCustomValue) soap.HasFault {
	body := &methods.SetCustomValueBody{}

	cfm := Map.CustomFieldsManager()

	_, field := cfm.findByNameType(req.Key, req.This.Type)
	if field == nil {
		body.Fault_ = Fault("", &types.InvalidArgument{InvalidProperty: "key"})
		return body
	}

	res := cfm.SetField(ctx, &types.SetField{
		This:   cfm.Reference(),
		Entity: req.This,
		Key:    field.Key,
		Value:  req.Value,
	})

	if res.Fault() != nil {
		body.Fault_ = res.Fault()
		return body
	}

	body.Res = &types.SetCustomValueResponse{}
	return body
}

// newUUID returns a stable UUID string based on input s
func newUUID(s string) string {
	return sha1UUID(s).String()
}

// sha1UUID returns a stable UUID based on input s
func sha1UUID(s string) uuid.UUID {
	return uuid.NewSHA1(uuid.NameSpaceOID, []byte(s))
}

// deepCopy uses xml encode/decode to copy src to dst
func deepCopy(src, dst interface{}) {
	b, err := xml.Marshal(src)
	if err != nil {
		panic(err)
	}

	dec := xml.NewDecoder(bytes.NewReader(b))
	dec.TypeFunc = types.TypeFunc()
	err = dec.Decode(dst)
	if err != nil {
		panic(err)
	}
}
