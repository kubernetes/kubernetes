// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package client

import (
	"fmt"
	"mime"
	"net/http"
	"net/http/httputil"
	"os"
	"path"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/net/context/ctxhttp"

	"github.com/go-openapi/runtime"
	"github.com/go-openapi/strfmt"
)

// DefaultTimeout the default request timeout
var DefaultTimeout = 30 * time.Second

// Runtime represents an API client that uses the transport
// to make http requests based on a swagger specification.
type Runtime struct {
	DefaultMediaType      string
	DefaultAuthentication runtime.ClientAuthInfoWriter
	Consumers             map[string]runtime.Consumer
	Producers             map[string]runtime.Producer

	Transport http.RoundTripper
	Jar       http.CookieJar
	//Spec      *spec.Document
	Host     string
	BasePath string
	Formats  strfmt.Registry
	Debug    bool
	Context  context.Context

	clientOnce *sync.Once
	client     *http.Client
	schemes    []string
}

// New creates a new default runtime for a swagger api runtime.Client
func New(host, basePath string, schemes []string) *Runtime {
	var rt Runtime
	rt.DefaultMediaType = runtime.JSONMime

	// TODO: actually infer this stuff from the spec
	rt.Consumers = map[string]runtime.Consumer{
		runtime.JSONMime: runtime.JSONConsumer(),
		runtime.XMLMime:  runtime.XMLConsumer(),
		runtime.TextMime: runtime.TextConsumer(),
	}
	rt.Producers = map[string]runtime.Producer{
		runtime.JSONMime: runtime.JSONProducer(),
		runtime.XMLMime:  runtime.XMLProducer(),
		runtime.TextMime: runtime.TextProducer(),
	}
	rt.Transport = http.DefaultTransport
	rt.Jar = nil
	rt.Host = host
	rt.BasePath = basePath
	rt.Context = context.Background()
	rt.clientOnce = new(sync.Once)
	if !strings.HasPrefix(rt.BasePath, "/") {
		rt.BasePath = "/" + rt.BasePath
	}
	rt.Debug = os.Getenv("DEBUG") == "1"
	if len(schemes) > 0 {
		rt.schemes = schemes
	}
	return &rt
}

func (r *Runtime) pickScheme(schemes []string) string {
	if v := r.selectScheme(r.schemes); v != "" {
		return v
	}
	if v := r.selectScheme(schemes); v != "" {
		return v
	}
	return "http"
}

func (r *Runtime) selectScheme(schemes []string) string {
	schLen := len(schemes)
	if schLen == 0 {
		return ""
	}

	scheme := schemes[0]
	// prefer https, but skip when not possible
	if scheme != "https" && schLen > 1 {
		for _, sch := range schemes {
			if sch == "https" {
				scheme = sch
				break
			}
		}
	}
	return scheme
}

// Submit a request and when there is a body on success it will turn that into the result
// all other things are turned into an api error for swagger which retains the status code
func (r *Runtime) Submit(operation *runtime.ClientOperation) (interface{}, error) {
	params, readResponse, auth := operation.Params, operation.Reader, operation.AuthInfo

	request, err := newRequest(operation.Method, operation.PathPattern, params)
	if err != nil {
		return nil, err
	}

	var accept []string
	for _, mimeType := range operation.ProducesMediaTypes {
		accept = append(accept, mimeType)
	}
	request.SetHeaderParam(runtime.HeaderAccept, accept...)

	if auth == nil && r.DefaultAuthentication != nil {
		auth = r.DefaultAuthentication
	}
	if auth != nil {
		if err := auth.AuthenticateRequest(request, r.Formats); err != nil {
			return nil, err
		}
	}

	// TODO: pick appropriate media type
	cmt := r.DefaultMediaType
	if len(operation.ConsumesMediaTypes) > 0 {
		cmt = operation.ConsumesMediaTypes[0]
	}

	req, err := request.BuildHTTP(cmt, r.Producers, r.Formats)
	if err != nil {
		return nil, err
	}
	req.URL.Scheme = r.pickScheme(operation.Schemes)
	req.URL.Host = r.Host
	var reinstateSlash bool
	if req.URL.Path != "" && req.URL.Path != "/" && req.URL.Path[len(req.URL.Path)-1] == '/' {
		reinstateSlash = true
	}
	req.URL.Path = path.Join(r.BasePath, req.URL.Path)
	if reinstateSlash {
		req.URL.Path = req.URL.Path + "/"
	}

	r.clientOnce.Do(func() {
		r.client = &http.Client{
			Transport: r.Transport,
			Jar:       r.Jar,
		}
	})

	if r.Debug {
		b, err2 := httputil.DumpRequestOut(req, true)
		if err2 != nil {
			return nil, err2
		}
		fmt.Println(string(b))
	}

	pctx := r.Context
	if pctx == nil {
		pctx = context.Background()
	}
	ctx, cancel := context.WithTimeout(pctx, request.timeout)
	defer cancel()

	res, err := ctxhttp.Do(ctx, r.client, req) // make requests, by default follows 10 redirects before failing
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if r.Debug {
		b, err2 := httputil.DumpResponse(res, true)
		if err2 != nil {
			return nil, err2
		}
		fmt.Println(string(b))
	}

	ct := res.Header.Get(runtime.HeaderContentType)
	if ct == "" { // this should really really never occur
		ct = r.DefaultMediaType
	}

	mt, _, err := mime.ParseMediaType(ct)
	if err != nil {
		return nil, fmt.Errorf("parse content type: %s", err)
	}

	cons, ok := r.Consumers[mt]
	if !ok {
		// scream about not knowing what to do
		return nil, fmt.Errorf("no consumer: %q", ct)
	}
	return readResponse.ReadResponse(response{res}, cons)
}
