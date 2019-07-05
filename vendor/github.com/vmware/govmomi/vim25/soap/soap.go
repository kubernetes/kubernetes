/*
Copyright (c) 2014-2018 VMware, Inc. All Rights Reserved.

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

package soap

import (
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vim25/xml"
)

// Header includes optional soap Header fields.
type Header struct {
	Action   string      `xml:"-"`                         // Action is the 'SOAPAction' HTTP header value. Defaults to "Client.Namespace/Client.Version".
	Cookie   string      `xml:"vcSessionCookie,omitempty"` // Cookie is a vCenter session cookie that can be used with other SDK endpoints (e.g. pbm).
	ID       string      `xml:"operationID,omitempty"`     // ID is the operationID used by ESX/vCenter logging for correlation.
	Security interface{} `xml:",omitempty"`                // Security is used for SAML token authentication and request signing.
}

type Envelope struct {
	XMLName xml.Name `xml:"http://schemas.xmlsoap.org/soap/envelope/ Envelope"`
	Header  *Header  `xml:"http://schemas.xmlsoap.org/soap/envelope/ Header,omitempty"`
	Body    interface{}
}

type Fault struct {
	XMLName xml.Name `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault"`
	Code    string   `xml:"faultcode"`
	String  string   `xml:"faultstring"`
	Detail  struct {
		Fault types.AnyType `xml:",any,typeattr"`
	} `xml:"detail"`
}

func (f *Fault) VimFault() types.AnyType {
	return f.Detail.Fault
}
