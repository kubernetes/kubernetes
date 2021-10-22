/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package types

import (
	"reflect"

	"github.com/vmware/govmomi/vim25/types"
)

// Types here are not included in the wsdl

const (
	RoleActAsUser     = "ActAsUser"
	RoleRegularUser   = "RegularUser"
	RoleAdministrator = "Administrator"
)

type GrantWSTrustRole GrantWSTrustRoleRequestType

func init() {
	types.Add("sso:GrantWSTrustRole", reflect.TypeOf((*GrantWSTrustRole)(nil)).Elem())
}

type GrantWSTrustRoleRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	UserId PrincipalId                  `xml:"userId"`
	Role   string                       `xml:"role"`
}

func init() {
	types.Add("sso:GrantWSTrustRoleRequestType", reflect.TypeOf((*GrantWSTrustRoleRequestType)(nil)).Elem())
}

type GrantWSTrustRoleResponse struct {
	Returnval bool `xml:"returnval"`
}

type RevokeWSTrustRole RevokeWSTrustRoleRequestType

func init() {
	types.Add("sso:RevokeWSTrustRole", reflect.TypeOf((*RevokeWSTrustRole)(nil)).Elem())
}

type RevokeWSTrustRoleRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	UserId PrincipalId                  `xml:"userId"`
	Role   string                       `xml:"role"`
}

func init() {
	types.Add("sso:RevokeWSTrustRoleRequestType", reflect.TypeOf((*RevokeWSTrustRoleRequestType)(nil)).Elem())
}

type RevokeWSTrustRoleResponse struct {
	Returnval bool `xml:"returnval"`
}
