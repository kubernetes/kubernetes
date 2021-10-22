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

package simulator

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"reflect"
	"testing"
	"time"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/simulator/vpx"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

func TestUnmarshal(t *testing.T) {
	requests := []struct {
		body interface{}
		data string
	}{
		{
			&types.RetrieveServiceContent{
				This: types.ManagedObjectReference{
					Type: "ServiceInstance", Value: "ServiceInstance",
				},
			},
			`<?xml version="1.0" encoding="UTF-8"?>
                         <Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/">
                           <Body>
                             <RetrieveServiceContent xmlns="urn:vim25">
                               <_this type="ServiceInstance">ServiceInstance</_this>
                             </RetrieveServiceContent>
                           </Body>
                         </Envelope>`,
		},
		{
			&types.Login{
				This: types.ManagedObjectReference{
					Type:  "SessionManager",
					Value: "SessionManager",
				},
				UserName: "root",
				Password: "secret",
			},
			`<?xml version="1.0" encoding="UTF-8"?>
                         <Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/">
                           <Body>
                             <Login xmlns="urn:vim25">
                               <_this type="SessionManager">SessionManager</_this>
                               <userName>root</userName>
                               <password>secret</password>
                             </Login>
                           </Body>
                         </Envelope>`,
		},
		{
			&types.RetrieveProperties{
				This: types.ManagedObjectReference{Type: "PropertyCollector", Value: "ha-property-collector"},
				SpecSet: []types.PropertyFilterSpec{
					{
						DynamicData: types.DynamicData{},
						PropSet: []types.PropertySpec{
							{
								DynamicData: types.DynamicData{},
								Type:        "ManagedEntity",
								All:         (*bool)(nil),
								PathSet:     []string{"name", "parent"},
							},
						},
						ObjectSet: []types.ObjectSpec{
							{
								DynamicData: types.DynamicData{},
								Obj:         types.ManagedObjectReference{Type: "Folder", Value: "ha-folder-root"},
								Skip:        types.NewBool(false),
								SelectSet: []types.BaseSelectionSpec{ // test decode of interface
									&types.TraversalSpec{
										SelectionSpec: types.SelectionSpec{
											DynamicData: types.DynamicData{},
											Name:        "traverseParent",
										},
										Type: "ManagedEntity",
										Path: "parent",
										Skip: types.NewBool(false),
										SelectSet: []types.BaseSelectionSpec{
											&types.SelectionSpec{
												DynamicData: types.DynamicData{},
												Name:        "traverseParent",
											},
										},
									},
								},
							},
						},
						ReportMissingObjectsInResults: (*bool)(nil),
					},
				}},
			`<?xml version="1.0" encoding="UTF-8"?>
                         <Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/">
                          <Body>
                           <RetrieveProperties xmlns="urn:vim25">
                            <_this type="PropertyCollector">ha-property-collector</_this>
                            <specSet>
                             <propSet>
                              <type>ManagedEntity</type>
                              <pathSet>name</pathSet>
                              <pathSet>parent</pathSet>
                             </propSet>
                             <objectSet>
                              <obj type="Folder">ha-folder-root</obj>
                              <skip>false</skip>
                              <selectSet xmlns:XMLSchema-instance="http://www.w3.org/2001/XMLSchema-instance" XMLSchema-instance:type="TraversalSpec">
                               <name>traverseParent</name>
                               <type>ManagedEntity</type>
                               <path>parent</path>
                               <skip>false</skip>
                               <selectSet XMLSchema-instance:type="SelectionSpec">
                                <name>traverseParent</name>
                               </selectSet>
                              </selectSet>
                             </objectSet>
                            </specSet>
                           </RetrieveProperties>
                          </Body>
                         </Envelope>`,
		},
	}

	for i, req := range requests {
		method, err := UnmarshalBody(vim25MapType, []byte(req.data))
		if err != nil {
			t.Errorf("failed to decode %d (%s): %s", i, req, err)
		}
		if !reflect.DeepEqual(method.Body, req.body) {
			t.Errorf("malformed body %d (%#v):", i, method.Body)
		}
	}
}

func TestUnmarshalError(t *testing.T) {
	requests := []string{
		"", // io.EOF
		`<?xml version="1.0" encoding="UTF-8"?>
                 <Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/">
                   <Body>
                   </MissingEndTag
                 </Envelope>`,
		`<?xml version="1.0" encoding="UTF-8"?>
                 <Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/">
                   <Body>
                     <UnknownType xmlns="urn:vim25">
                       <_this type="ServiceInstance">ServiceInstance</_this>
                     </UnknownType>
                   </Body>
                 </Envelope>`,
		`<?xml version="1.0" encoding="UTF-8"?>
                 <Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/">
                   <Body>
                   <!-- no start tag -->
                   </Body>
                 </Envelope>`,
		`<?xml version="1.0" encoding="UTF-8"?>
                 <Envelope xmlns="http://schemas.xmlsoap.org/soap/envelope/">
                   <Body>
                     <NoSuchMethod xmlns="urn:vim25">
                       <_this type="ServiceInstance">ServiceInstance</_this>
                     </NoSuchMethod>
                   </Body>
                 </Envelope>`,
	}

	for i, data := range requests {
		if _, err := UnmarshalBody(vim25MapType, []byte(data)); err != nil {
			continue
		}
		t.Errorf("expected %d (%s) to return an error", i, data)
	}
}

func TestServeHTTP(t *testing.T) {
	configs := []struct {
		content types.ServiceContent
		folder  mo.Folder
	}{
		{esx.ServiceContent, esx.RootFolder},
		{vpx.ServiceContent, vpx.RootFolder},
	}

	for _, config := range configs {
		s := New(NewServiceInstance(config.content, config.folder))
		ts := s.NewServer()
		defer ts.Close()

		u := ts.URL.User
		ts.URL.User = nil

		ctx := context.Background()
		client, err := govmomi.NewClient(ctx, ts.URL, true)
		if err != nil {
			t.Fatal(err)
		}

		err = client.Login(ctx, nil)
		if err == nil {
			t.Fatal("expected invalid login error")
		}

		err = client.Login(ctx, u)
		if err != nil {
			t.Fatal(err)
		}

		// Testing http client + reflect client
		clients := []soap.RoundTripper{client, s.client}
		for _, c := range clients {
			now, err := methods.GetCurrentTime(ctx, c)
			if err != nil {
				t.Fatal(err)
			}

			if now.After(time.Now()) {
				t.Fail()
			}

			// test the fail/Fault path
			_, err = methods.QueryVMotionCompatibility(ctx, c, &types.QueryVMotionCompatibility{})
			if err == nil {
				t.Errorf("expected error")
			}
		}

		err = client.Logout(ctx)
		if err != nil {
			t.Error(err)
		}
	}
}

func TestServeAbout(t *testing.T) {
	ctx := context.Background()

	m := VPX()
	m.App = 1
	m.Pod = 1

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	u := *s.URL
	u.Path += "/vimServiceVersions.xml"
	r, err := c.Get(u.String())
	if err != nil {
		t.Fatal(err)
	}
	_ = r.Body.Close()

	u.Path = "/about"
	r, err = c.Get(u.String())
	if err != nil {
		t.Fatal(err)
	}
	_ = r.Body.Close()
}

func TestServeHTTPS(t *testing.T) {
	s := New(NewServiceInstance(esx.ServiceContent, esx.RootFolder))
	s.TLS = new(tls.Config)
	ts := s.NewServer()
	defer ts.Close()

	ts.Config.ErrorLog = log.New(ioutil.Discard, "", 0) // silence benign "TLS handshake error" log messages

	ctx := context.Background()

	// insecure=true OK
	_, err := govmomi.NewClient(ctx, ts.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	// insecure=false should FAIL
	_, err = govmomi.NewClient(ctx, ts.URL, false)
	if err == nil {
		t.Fatal("expected error")
	}

	uerr, ok := err.(*url.Error)
	if !ok {
		t.Fatalf("err type=%T", err)
	}

	_, ok = uerr.Err.(x509.UnknownAuthorityError)
	if !ok {
		t.Fatalf("err type=%T", uerr.Err)
	}

	sinfo := ts.CertificateInfo()

	// Test thumbprint validation
	sc := soap.NewClient(ts.URL, false)
	// Add host with thumbprint mismatch should fail
	sc.SetThumbprint(ts.URL.Host, "nope")
	_, err = vim25.NewClient(ctx, sc)
	if err == nil {
		t.Error("expected error")
	}
	// Add host with thumbprint match should pass
	sc.SetThumbprint(ts.URL.Host, sinfo.ThumbprintSHA1)
	_, err = vim25.NewClient(ctx, sc)
	if err != nil {
		t.Fatal(err)
	}

	var pinfo object.HostCertificateInfo
	err = pinfo.FromURL(ts.URL, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pinfo.ThumbprintSHA1 != sinfo.ThumbprintSHA1 {
		t.Error("thumbprint mismatch")
	}

	// Test custom RootCAs list
	sc = soap.NewClient(ts.URL, false)
	caFile, err := ts.CertificateFile()
	if err != nil {
		t.Fatal(err)
	}
	if err = sc.SetRootCAs(caFile); err != nil {
		t.Fatal(err)
	}

	_, err = vim25.NewClient(ctx, sc)
	if err != nil {
		t.Fatal(err)
	}
}

type errorMarshal struct {
	mo.ServiceInstance
}

func (*errorMarshal) Fault() *soap.Fault {
	return nil
}

func (*errorMarshal) MarshalText() ([]byte, error) {
	return nil, errors.New("time has stopped")
}

func (h *errorMarshal) CurrentTime(types.AnyType) soap.HasFault {
	return h
}

type errorNoSuchMethod struct {
	mo.ServiceInstance
}

func TestServeHTTPErrors(t *testing.T) {
	s := New(NewServiceInstance(esx.ServiceContent, esx.RootFolder))

	ts := s.NewServer()
	defer ts.Close()

	ctx := context.Background()
	client, err := govmomi.NewClient(ctx, ts.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	// test response to unimplemented method
	req := &types.QueryMemoryOverhead{This: esx.HostSystem.Reference()}
	_, err = methods.QueryMemoryOverhead(ctx, client.Client, req)
	if _, ok := soap.ToSoapFault(err).VimFault().(types.MethodNotFound); !ok {
		t.Error("expected MethodNotFound fault")
	}

	// cover the does not implement method error path
	Map.objects[vim25.ServiceInstance] = &errorNoSuchMethod{}
	_, err = methods.GetCurrentTime(ctx, client)
	if err == nil {
		t.Error("expected error")
	}

	// cover the xml encode error path
	Map.objects[vim25.ServiceInstance] = &errorMarshal{}
	_, err = methods.GetCurrentTime(ctx, client)
	if err == nil {
		t.Error("expected error")
	}

	// cover the no such object path
	Map.Remove(vim25.ServiceInstance)
	_, err = methods.GetCurrentTime(ctx, client)
	if err == nil {
		t.Error("expected error")
	}

	// verify we properly marshal the fault
	fault := soap.ToSoapFault(err).VimFault()
	f, ok := fault.(types.ManagedObjectNotFound)
	if !ok {
		t.Fatalf("fault=%#v", fault)
	}
	if f.Obj != vim25.ServiceInstance {
		t.Errorf("obj=%#v", f.Obj)
	}

	// cover the method not supported path
	res, err := http.Get(ts.URL.String())
	if err != nil {
		log.Fatal(err)
	}

	if res.StatusCode != http.StatusMethodNotAllowed {
		t.Errorf("expected status %d, got %s", http.StatusMethodNotAllowed, res.Status)
	}

	// cover the ioutil.ReadAll error path
	s.readAll = func(io.Reader) ([]byte, error) {
		return nil, io.ErrShortBuffer
	}
	res, err = http.Post(ts.URL.String(), "none", nil)
	if err != nil {
		log.Fatal(err)
	}

	if res.StatusCode != http.StatusBadRequest {
		t.Errorf("expected status %d, got %s", http.StatusBadRequest, res.Status)
	}
}
