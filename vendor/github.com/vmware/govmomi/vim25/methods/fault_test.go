/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package methods

import (
	"bytes"
	"testing"

	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vim25/xml"
)

var invalidLoginFault = `
<?xml version="1.0" encoding="UTF-8"?>
<soapenv:Envelope xmlns:soapenc="http://schemas.xmlsoap.org/soap/encoding/"
 xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"
 xmlns:xsd="http://www.w3.org/2001/XMLSchema"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<soapenv:Body>
<soapenv:Fault><faultcode>ServerFaultCode</faultcode><faultstring>Cannot complete login due to an incorrect user name or password.</faultstring><detail><InvalidLoginFault xmlns="urn:vim25" xsi:type="InvalidLogin"></InvalidLoginFault></detail></soapenv:Fault>
</soapenv:Body>
</soapenv:Envelope>`

type TestBody struct {
	Fault *soap.Fault `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func TestFaultDetail(t *testing.T) {
	body := TestBody{}
	env := soap.Envelope{Body: &body}

	dec := xml.NewDecoder(bytes.NewReader([]byte(invalidLoginFault)))
	dec.TypeFunc = types.TypeFunc()

	err := dec.Decode(&env)
	if err != nil {
		t.Fatalf("Decode: %s", err)
	}

	if body.Fault == nil {
		t.Fatal("Expected fault")
	}

	if _, ok := body.Fault.Detail.Fault.(types.InvalidLogin); !ok {
		t.Fatalf("Expected InvalidLogin, got: %#v", body.Fault.Detail.Fault)
	}
}
