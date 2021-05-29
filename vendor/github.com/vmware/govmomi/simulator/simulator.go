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
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"

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
	Name   string
	This   types.ManagedObjectReference
	Header soap.Header
	Body   types.AnyType
}

// Service decodes incoming requests and dispatches to a Handler
type Service struct {
	client *vim25.Client
	sm     *SessionManager
	sdk    map[string]*Registry
	delay  *DelayConfig

	readAll func(io.Reader) ([]byte, error)

	TLS      *tls.Config
	ServeMux *http.ServeMux
}

// Server provides a simulator Service over HTTP
type Server struct {
	*httptest.Server
	URL    *url.URL
	Tunnel int

	caFile string
}

// New returns an initialized simulator Service instance
func New(instance *ServiceInstance) *Service {
	s := &Service{
		readAll: ioutil.ReadAll,
		sm:      Map.SessionManager(),
		sdk:     make(map[string]*Registry),
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

func (s *Service) call(ctx *Context, method *Method) soap.HasFault {
	handler := ctx.Map.Get(method.This)
	session := ctx.Session

	if session == nil {
		switch method.Name {
		case "RetrieveServiceContent", "PbmRetrieveServiceContent", "List", "Login", "LoginByToken", "LoginExtensionByCertificate", "RetrieveProperties", "RetrievePropertiesEx", "CloneSession":
			// ok for now, TODO: authz
		default:
			fault := &types.NotAuthenticated{
				NoPermission: types.NoPermission{
					Object:      method.This,
					PrivilegeId: "System.View",
				},
			}
			return &serverFaultBody{Reason: Fault("", fault)}
		}
	} else {
		// Prefer the Session.Registry, ServiceContent.PropertyCollector filter field for example is per-session
		if h := session.Get(method.This); h != nil {
			handler = h
		}
	}

	if handler == nil {
		msg := fmt.Sprintf("managed object not found: %s", method.This)
		log.Print(msg)
		fault := &types.ManagedObjectNotFound{Obj: method.This}
		return &serverFaultBody{Reason: Fault(msg, fault)}
	}

	// Lowercase methods can't be accessed outside their package
	name := strings.Title(method.Name)

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

	// We have a valid call. Introduce a delay if requested
	//
	if s.delay != nil {
		d := 0
		if s.delay.Delay > 0 {
			d = s.delay.Delay
		}
		if md, ok := s.delay.MethodDelay[method.Name]; ok {
			d += md
		}
		if s.delay.DelayJitter > 0 {
			d += int(rand.NormFloat64() * s.delay.DelayJitter * float64(d))
		}
		if d > 0 {
			//fmt.Printf("Delaying method %s %d ms\n", name, d)
			time.Sleep(time.Duration(d) * time.Millisecond)
		}
	}

	var args, res []reflect.Value
	if m.Type().NumIn() == 2 {
		args = append(args, reflect.ValueOf(ctx))
	}
	args = append(args, reflect.ValueOf(method.Body))
	ctx.Map.WithLock(handler, func() {
		res = m.Call(args)
	})

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

	res := s.call(&Context{
		Map:     Map,
		Context: ctx,
		Session: internalContext.Session,
	}, method)

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

type faultDetail struct {
	Fault types.AnyType
}

// soapFault is a copy of soap.Fault, with the same changes as soapEnvelope
type soapFault struct {
	XMLName xml.Name `xml:"soapenv:Fault"`
	Code    string   `xml:"faultcode"`
	String  string   `xml:"faultstring"`
	Detail  struct {
		Fault *faultDetail
	} `xml:"detail"`
}

// MarshalXML renames the start element from "Fault" to "${Type}Fault"
func (d *faultDetail) MarshalXML(e *xml.Encoder, start xml.StartElement) error {
	kind := reflect.TypeOf(d.Fault).Elem().Name()
	start.Name.Local = kind + "Fault"
	start.Attr = append(start.Attr,
		xml.Attr{
			Name:  xml.Name{Local: "xmlns"},
			Value: "urn:" + vim25.Namespace,
		},
		xml.Attr{
			Name:  xml.Name{Local: "xsi:type"},
			Value: kind,
		})
	return e.EncodeElement(d.Fault, start)
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

			in := m.Type.NumIn()
			if in < 2 || in > 3 { // at least 2 params (receiver and request), optionally a 3rd param (context)
				continue
			}
			if m.Type.NumOut() != 1 || m.Type.Out(0) != f { // all methods return soap.HasFault
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

// Handle registers the handler for the given pattern with Service.ServeMux.
func (s *Service) Handle(pattern string, handler http.Handler) {
	s.ServeMux.Handle(pattern, handler)
	// Not ideal, but avoids having to add yet another registration mechanism
	// so we can optionally use vapi/simulator internally.
	if m, ok := handler.(tagManager); ok {
		s.sdk[vim25.Path].tagManager = m
	}
}

// RegisterSDK adds an HTTP handler for the Registry's Path and Namespace.
func (s *Service) RegisterSDK(r *Registry) {
	if s.ServeMux == nil {
		s.ServeMux = http.NewServeMux()
	}

	s.sdk[r.Path] = r
	s.ServeMux.HandleFunc(r.Path, s.ServeSDK)
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

	ctx := &Context{
		req: r,
		res: w,
		svc: s,

		Map:     s.sdk[r.URL.Path],
		Context: context.Background(),
	}
	ctx.Map.WithLock(s.sm, ctx.mapSession)

	var res soap.HasFault
	var soapBody interface{}

	method, err := UnmarshalBody(ctx.Map.typeFunc, body)
	if err != nil {
		res = serverFault(err.Error())
	} else {
		ctx.Header = method.Header
		res = s.call(ctx, method)
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
				Detail: struct {
					Fault *faultDetail
				}{&faultDetail{f.Detail.Fault}},
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
	dc, err := finder.DatacenterOrDefault(ctx, query.Get("dcPath"))
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

	r.URL.Path = strings.TrimPrefix(r.URL.Path, folderPrefix)
	p := path.Join(ds.Info.GetDatastoreInfo().Url, r.URL.Path)

	switch r.Method {
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
		dir := path.Dir(p)
		_ = os.MkdirAll(dir, 0700)

		f, err := os.Create(p)
		if err != nil {
			log.Printf("failed to %s '%s': %s", r.Method, p, err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		defer f.Close()

		_, _ = io.Copy(f, r.Body)
	default:
		fs := http.FileServer(http.Dir(ds.Info.GetDatastoreInfo().Url))

		fs.ServeHTTP(w, r)
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

// defaultIP returns addr.IP if specified, otherwise attempts to find a non-loopback ipv4 IP
func defaultIP(addr *net.TCPAddr) string {
	if !addr.IP.IsUnspecified() {
		return addr.IP.String()
	}

	nics, err := net.Interfaces()
	if err != nil {
		return addr.IP.String()
	}

	for _, nic := range nics {
		if nic.Name == "docker0" || strings.HasPrefix(nic.Name, "vmnet") {
			continue
		}
		addrs, aerr := nic.Addrs()
		if aerr != nil {
			continue
		}
		for _, addr := range addrs {
			if ip, ok := addr.(*net.IPNet); ok && !ip.IP.IsLoopback() {
				if ip.IP.To4() != nil {
					return ip.IP.String()
				}
			}
		}
	}

	return addr.IP.String()
}

// NewServer returns an http Server instance for the given service
func (s *Service) NewServer() *Server {
	s.RegisterSDK(Map)

	mux := s.ServeMux
	mux.HandleFunc(Map.Path+"/vimServiceVersions.xml", s.ServiceVersions)
	mux.HandleFunc(folderPrefix, s.ServeDatastore)
	mux.HandleFunc("/about", s.About)

	// Using NewUnstartedServer() instead of NewServer(),
	// for use in main.go, where Start() blocks, we can still set ServiceHostName
	ts := httptest.NewUnstartedServer(mux)

	addr := ts.Listener.Addr().(*net.TCPAddr)
	port := strconv.Itoa(addr.Port)
	u := &url.URL{
		Scheme: "http",
		Host:   net.JoinHostPort(defaultIP(addr), port),
		Path:   Map.Path,
	}

	// Redirect clients to this http server, rather than HostSystem.Name
	Map.SessionManager().ServiceHostName = u.Host

	if f := flag.Lookup("httptest.serve"); f != nil {
		// Avoid the blocking behaviour of httptest.Server.Start() when this flag is set
		_ = f.Value.Set("")
	}

	cert := ""
	if s.TLS == nil {
		ts.Start()
	} else {
		ts.TLS = s.TLS
		ts.TLS.ClientAuth = tls.RequestClientCert // Used by SessionManager.LoginExtensionByCertificate
		ts.StartTLS()
		u.Scheme += "s"

		cert = base64.StdEncoding.EncodeToString(ts.TLS.Certificates[0].Certificate[0])
	}

	// Add vcsim config to OptionManager for use by SDK handlers (see lookup/simulator for example)
	m := Map.OptionManager()
	m.Setting = append(m.Setting,
		&types.OptionValue{
			Key:   "vcsim.server.url",
			Value: u.String(),
		},
		&types.OptionValue{
			Key:   "vcsim.server.cert",
			Value: cert,
		},
	)

	u.User = url.UserPassword("user", "pass")

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

// proxy tunnels SDK requests
func (s *Server) proxy(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodConnect {
		http.Error(w, "", http.StatusMethodNotAllowed)
		return
	}

	dst, err := net.Dial("tcp", s.URL.Host)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.WriteHeader(http.StatusOK)

	src, _, err := w.(http.Hijacker).Hijack()
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	go io.Copy(src, dst)
	go func() {
		_, _ = io.Copy(dst, src)
		_ = dst.Close()
		_ = src.Close()
	}()
}

// StartTunnel runs an HTTP proxy for tunneling SDK requests that require TLS client certificate authentication.
func (s *Server) StartTunnel() error {
	tunnel := &http.Server{
		Addr:    fmt.Sprintf("%s:%d", s.URL.Hostname(), s.Tunnel),
		Handler: http.HandlerFunc(s.proxy),
	}

	l, err := net.Listen("tcp", tunnel.Addr)
	if err != nil {
		return err
	}

	if s.Tunnel == 0 {
		s.Tunnel = l.Addr().(*net.TCPAddr).Port
	}

	// Set client proxy port (defaults to vCenter host port 80 in real life)
	q := s.URL.Query()
	q.Set("GOVMOMI_TUNNEL_PROXY_PORT", strconv.Itoa(s.Tunnel))
	s.URL.RawQuery = q.Encode()

	go tunnel.Serve(l)

	return nil
}

// Close shuts down the server and blocks until all outstanding
// requests on this server have completed.
func (s *Server) Close() {
	s.Server.Close()
	if s.caFile != "" {
		_ = os.Remove(s.caFile)
	}
}

var (
	vim25MapType = types.TypeFunc()
)

func defaultMapType(name string) (reflect.Type, bool) {
	typ, ok := vim25MapType(name)
	if !ok {
		// See TestIssue945, in which case Go does not resolve the namespace and name == "ns1:TraversalSpec"
		// Without this hack, the SelectSet would be all nil's
		kind := strings.SplitN(name, ":", 2)
		if len(kind) == 2 {
			typ, ok = vim25MapType(kind[1])
		}
	}
	return typ, ok
}

// Element can be used to defer decoding of an XML node.
type Element struct {
	start xml.StartElement
	inner struct {
		Content string `xml:",innerxml"`
	}
	typeFunc func(string) (reflect.Type, bool)
}

func (e *Element) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	e.start = start

	return d.DecodeElement(&e.inner, &start)
}

func (e *Element) decoder() *xml.Decoder {
	decoder := xml.NewDecoder(strings.NewReader(e.inner.Content))
	decoder.TypeFunc = e.typeFunc // required to decode interface types
	return decoder
}

func (e *Element) Decode(val interface{}) error {
	return e.decoder().DecodeElement(val, &e.start)
}

// UnmarshalBody extracts the Body from a soap.Envelope and unmarshals to the corresponding govmomi type
func UnmarshalBody(typeFunc func(string) (reflect.Type, bool), data []byte) (*Method, error) {
	body := &Element{typeFunc: typeFunc}
	req := soap.Envelope{
		Header: &soap.Header{
			Security: new(Element),
		},
		Body: body,
	}

	err := xml.Unmarshal(data, &req)
	if err != nil {
		return nil, fmt.Errorf("xml.Unmarshal: %s", err)
	}

	var start xml.StartElement
	var ok bool
	decoder := body.decoder()

	for {
		tok, derr := decoder.Token()
		if derr != nil {
			return nil, fmt.Errorf("decoding: %s", derr)
		}
		if start, ok = tok.(xml.StartElement); ok {
			break
		}
	}

	if !ok {
		return nil, fmt.Errorf("decoding: method token not found")
	}

	kind := start.Name.Local
	rtype, ok := typeFunc(kind)
	if !ok {
		return nil, fmt.Errorf("no vmomi type defined for '%s'", kind)
	}

	val := reflect.New(rtype).Interface()

	err = decoder.DecodeElement(val, &start)
	if err != nil {
		return nil, fmt.Errorf("decoding %s: %s", kind, err)
	}

	method := &Method{Name: kind, Header: *req.Header, Body: val}

	field := reflect.ValueOf(val).Elem().FieldByName("This")

	method.This = field.Interface().(types.ManagedObjectReference)

	return method, nil
}
