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
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path"
	"reflect"
	"sort"
	"strings"

	"github.com/vmware/govmomi/find"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vim25/xml"
)

// Trace when set to true, writes SOAP traffic to stderr
var Trace = false

// Method encapsulates a decoded SOAP client request
type Method struct {
	Name string
	This types.ManagedObjectReference
	Body types.AnyType
}

// Service decodes incoming requests and dispatches to a Handler
type Service struct {
	client *vim25.Client

	readAll func(io.Reader) ([]byte, error)

	TLS *tls.Config
}

// Server provides a simulator Service over HTTP
type Server struct {
	*httptest.Server
	URL *url.URL

	caFile string
}

// New returns an initialized simulator Service instance
func New(instance *ServiceInstance) *Service {
	s := &Service{
		readAll: ioutil.ReadAll,
	}

	s.client, _ = vim25.NewClient(context.Background(), s)

	return s
}

type serverFaultBody struct {
	Reason *soap.Fault `xml:"http://schemas.xmlsoap.org/soap/envelope/ Fault,omitempty"`
}

func (b *serverFaultBody) Fault() *soap.Fault { return b.Reason }

func serverFault(msg string) soap.HasFault {
	return &serverFaultBody{Reason: Fault(msg, &types.InvalidRequest{})}
}

// Fault wraps the given message and fault in a soap.Fault
func Fault(msg string, fault types.BaseMethodFault) *soap.Fault {
	f := &soap.Fault{
		Code:   "ServerFaultCode",
		String: msg,
	}

	f.Detail.Fault = fault

	return f
}

func (s *Service) call(method *Method) soap.HasFault {
	handler := Map.Get(method.This)

	if handler == nil {
		msg := fmt.Sprintf("managed object not found: %s", method.This)
		log.Print(msg)
		fault := &types.ManagedObjectNotFound{Obj: method.This}
		return &serverFaultBody{Reason: Fault(msg, fault)}
	}

	name := method.Name

	if strings.HasSuffix(name, vTaskSuffix) {
		// Make golint happy renaming "Foo_Task" -> "FooTask"
		name = name[:len(name)-len(vTaskSuffix)] + sTaskSuffix
	}

	m := reflect.ValueOf(handler).MethodByName(name)
	if !m.IsValid() {
		msg := fmt.Sprintf("%s does not implement: %s", method.This, method.Name)
		log.Print(msg)
		fault := &types.MethodNotFound{Receiver: method.This, Method: method.Name}
		return &serverFaultBody{Reason: Fault(msg, fault)}
	}

	if e, ok := handler.(mo.Entity); ok {
		for _, dm := range e.Entity().DisabledMethod {
			if name == dm {
				msg := fmt.Sprintf("%s method is disabled: %s", method.This, method.Name)
				fault := &types.MethodDisabled{}
				return &serverFaultBody{Reason: Fault(msg, fault)}
			}
		}
	}

	res := m.Call([]reflect.Value{reflect.ValueOf(method.Body)})

	return res[0].Interface().(soap.HasFault)
}

// RoundTrip implements the soap.RoundTripper interface in process.
// Rather than encode/decode SOAP over HTTP, this implementation uses reflection.
func (s *Service) RoundTrip(ctx context.Context, request, response soap.HasFault) error {
	field := func(r soap.HasFault, name string) reflect.Value {
		return reflect.ValueOf(r).Elem().FieldByName(name)
	}

	// Every struct passed to soap.RoundTrip has "Req" and "Res" fields
	req := field(request, "Req")

	// Every request has a "This" field.
	this := req.Elem().FieldByName("This")

	method := &Method{
		Name: req.Elem().Type().Name(),
		This: this.Interface().(types.ManagedObjectReference),
		Body: req.Interface(),
	}

	res := s.call(method)

	if err := res.Fault(); err != nil {
		return soap.WrapSoapFault(err)
	}

	field(response, "Res").Set(field(res, "Res"))

	return nil
}

// soapEnvelope is a copy of soap.Envelope, with namespace changed to "soapenv",
// and additional namespace attributes required by some client libraries.
// Go still has issues decoding with such a namespace, but encoding is ok.
type soapEnvelope struct {
	XMLName xml.Name    `xml:"soapenv:Envelope"`
	Enc     string      `xml:"xmlns:soapenc,attr"`
	Env     string      `xml:"xmlns:soapenv,attr"`
	XSD     string      `xml:"xmlns:xsd,attr"`
	XSI     string      `xml:"xmlns:xsi,attr"`
	Body    interface{} `xml:"soapenv:Body"`
}

// soapFault is a copy of soap.Fault, with the same changes as soapEnvelope
type soapFault struct {
	XMLName xml.Name `xml:"soapenv:Fault"`
	Code    string   `xml:"faultcode"`
	String  string   `xml:"faultstring"`
	Detail  struct {
		Fault types.AnyType `xml:",any,typeattr"`
	} `xml:"detail"`
}

// About generates some info about the simulator.
func (s *Service) About(w http.ResponseWriter, r *http.Request) {
	var about struct {
		Methods []string
		Types   []string
	}

	seen := make(map[string]bool)

	f := reflect.TypeOf((*soap.HasFault)(nil)).Elem()

	for _, obj := range Map.objects {
		kind := obj.Reference().Type
		if seen[kind] {
			continue
		}
		seen[kind] = true

		about.Types = append(about.Types, kind)

		t := reflect.TypeOf(obj)
		for i := 0; i < t.NumMethod(); i++ {
			m := t.Method(i)
			if seen[m.Name] {
				continue
			}
			seen[m.Name] = true

			if m.Type.NumIn() != 2 || m.Type.NumOut() != 1 || m.Type.Out(0) != f {
				continue
			}

			about.Methods = append(about.Methods, strings.Replace(m.Name, "Task", "_Task", 1))
		}
	}

	sort.Strings(about.Methods)
	sort.Strings(about.Types)

	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(&about)
}

// ServeSDK implements the http.Handler interface
func (s *Service) ServeSDK(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	body, err := s.readAll(r.Body)
	_ = r.Body.Close()
	if err != nil {
		log.Printf("error reading body: %s", err)
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	if Trace {
		fmt.Fprintf(os.Stderr, "Request: %s\n", string(body))
	}

	var res soap.HasFault
	var soapBody interface{}

	method, err := UnmarshalBody(body)
	if err != nil {
		res = serverFault(err.Error())
	} else {
		res = s.call(method)
	}

	if f := res.Fault(); f != nil {
		w.WriteHeader(http.StatusInternalServerError)

		// the generated method/*Body structs use the '*soap.Fault' type,
		// so we need our own Body type to use the modified '*soapFault' type.
		soapBody = struct {
			Fault *soapFault
		}{
			&soapFault{
				Code:   f.Code,
				String: f.String,
				Detail: f.Detail,
			},
		}
	} else {
		w.WriteHeader(http.StatusOK)

		soapBody = res
	}

	var out bytes.Buffer

	fmt.Fprint(&out, xml.Header)
	e := xml.NewEncoder(&out)
	err = e.Encode(&soapEnvelope{
		Enc:  "http://schemas.xmlsoap.org/soap/encoding/",
		Env:  "http://schemas.xmlsoap.org/soap/envelope/",
		XSD:  "http://www.w3.org/2001/XMLSchema",
		XSI:  "http://www.w3.org/2001/XMLSchema-instance",
		Body: soapBody,
	})
	if err == nil {
		err = e.Flush()
	}

	if err != nil {
		log.Printf("error encoding %s response: %s", method.Name, err)
		return
	}

	if Trace {
		fmt.Fprintf(os.Stderr, "Response: %s\n", out.String())
	}

	_, _ = w.Write(out.Bytes())
}

func (s *Service) findDatastore(query url.Values) (*Datastore, error) {
	ctx := context.Background()

	finder := find.NewFinder(s.client, false)
	dc, err := finder.DatacenterOrDefault(ctx, query.Get("dcName"))
	if err != nil {
		return nil, err
	}

	finder.SetDatacenter(dc)

	ds, err := finder.DatastoreOrDefault(ctx, query.Get("dsName"))
	if err != nil {
		return nil, err
	}

	return Map.Get(ds.Reference()).(*Datastore), nil
}

const folderPrefix = "/folder/"

// ServeDatastore handler for Datastore access via /folder path.
func (s *Service) ServeDatastore(w http.ResponseWriter, r *http.Request) {
	ds, ferr := s.findDatastore(r.URL.Query())
	if ferr != nil {
		log.Printf("failed to locate datastore with query params: %s", r.URL.RawQuery)
		w.WriteHeader(http.StatusNotFound)
		return
	}

	file := strings.TrimPrefix(r.URL.Path, folderPrefix)
	p := path.Join(ds.Info.GetDatastoreInfo().Url, file)

	switch r.Method {
	case "GET":
		f, err := os.Open(p)
		if err != nil {
			log.Printf("failed to %s '%s': %s", r.Method, p, err)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		defer f.Close()

		_, _ = io.Copy(w, f)
	case "POST":
		_, err := os.Stat(p)
		if err == nil {
			// File exists
			w.WriteHeader(http.StatusConflict)
			return
		}

		// File does not exist, fallthrough to create via PUT logic
		fallthrough
	case "PUT":
		f, err := os.Create(p)
		if err != nil {
			log.Printf("failed to %s '%s': %s", r.Method, p, err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		defer f.Close()

		_, _ = io.Copy(f, r.Body)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

// ServiceVersions handler for the /sdk/vimServiceVersions.xml path.
func (*Service) ServiceVersions(w http.ResponseWriter, r *http.Request) {
	// pyvmomi depends on this

	const versions = xml.Header + `<namespaces version="1.0">
 <namespace>
  <name>urn:vim25</name>
  <version>6.5</version>
  <priorVersions>
   <version>6.0</version>
   <version>5.5</version>
  </priorVersions>
 </namespace>
</namespaces>
`
	fmt.Fprint(w, versions)
}

// NewServer returns an http Server instance for the given service
func (s *Service) NewServer() *Server {
	mux := http.NewServeMux()
	path := "/sdk"

	mux.HandleFunc(path, s.ServeSDK)
	mux.HandleFunc(path+"/vimServiceVersions.xml", s.ServiceVersions)
	mux.HandleFunc(folderPrefix, s.ServeDatastore)
	mux.HandleFunc("/about", s.About)

	// Using NewUnstartedServer() instead of NewServer(),
	// for use in main.go, where Start() blocks, we can still set ServiceHostName
	ts := httptest.NewUnstartedServer(mux)

	u := &url.URL{
		Scheme: "http",
		Host:   ts.Listener.Addr().String(),
		Path:   path,
		User:   url.UserPassword("user", "pass"),
	}

	// Redirect clients to this http server, rather than HostSystem.Name
	Map.Get(*s.client.ServiceContent.SessionManager).(*SessionManager).ServiceHostName = u.Host

	if f := flag.Lookup("httptest.serve"); f != nil {
		// Avoid the blocking behaviour of httptest.Server.Start() when this flag is set
		_ = f.Value.Set("")
	}

	if s.TLS == nil {
		ts.Start()
	} else {
		ts.TLS = s.TLS
		ts.StartTLS()
		u.Scheme += "s"
	}

	return &Server{
		Server: ts,
		URL:    u,
	}
}

// Certificate returns the TLS certificate for the Server if started with TLS enabled.
// This method will panic if TLS is not enabled for the server.
func (s *Server) Certificate() *x509.Certificate {
	// By default httptest.StartTLS uses http/internal.LocalhostCert, which we can access here:
	cert, _ := x509.ParseCertificate(s.TLS.Certificates[0].Certificate[0])
	return cert
}

// CertificateInfo returns Server.Certificate() as object.HostCertificateInfo
func (s *Server) CertificateInfo() *object.HostCertificateInfo {
	info := new(object.HostCertificateInfo)
	info.FromCertificate(s.Certificate())
	return info
}

// CertificateFile returns a file name, where the file contains the PEM encoded Server.Certificate.
// The temporary file is removed when Server.Close() is called.
func (s *Server) CertificateFile() (string, error) {
	if s.caFile != "" {
		return s.caFile, nil
	}

	f, err := ioutil.TempFile("", "vcsim-")
	if err != nil {
		return "", err
	}
	defer f.Close()

	s.caFile = f.Name()
	cert := s.Certificate()
	return s.caFile, pem.Encode(f, &pem.Block{Type: "CERTIFICATE", Bytes: cert.Raw})
}

// Close shuts down the server and blocks until all outstanding
// requests on this server have completed.
func (s *Server) Close() {
	s.Server.Close()
	if s.caFile != "" {
		_ = os.Remove(s.caFile)
	}
}

var typeFunc = types.TypeFunc()

// UnmarshalBody extracts the Body from a soap.Envelope and unmarshals to the corresponding govmomi type
func UnmarshalBody(data []byte) (*Method, error) {
	body := struct {
		Content string `xml:",innerxml"`
	}{}

	req := soap.Envelope{
		Body: &body,
	}

	err := xml.Unmarshal(data, &req)
	if err != nil {
		return nil, fmt.Errorf("xml.Unmarshal: %s", err)
	}

	decoder := xml.NewDecoder(bytes.NewReader([]byte(body.Content)))
	decoder.TypeFunc = typeFunc // required to decode interface types

	var start *xml.StartElement

	for {
		tok, derr := decoder.Token()
		if derr != nil {
			return nil, fmt.Errorf("decoding body: %s", err)
		}
		if t, ok := tok.(xml.StartElement); ok {
			start = &t
			break
		}
	}

	kind := start.Name.Local

	rtype, ok := typeFunc(kind)
	if !ok {
		return nil, fmt.Errorf("no vmomi type defined for '%s'", kind)
	}

	var val interface{}
	if rtype != nil {
		val = reflect.New(rtype).Interface()
	}

	err = decoder.DecodeElement(val, start)
	if err != nil {
		return nil, fmt.Errorf("decoding %s: %s", kind, err)
	}

	method := &Method{Name: kind, Body: val}

	field := reflect.ValueOf(val).Elem().FieldByName("This")

	method.This = field.Interface().(types.ManagedObjectReference)

	return method, nil
}
