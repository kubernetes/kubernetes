// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package remote_api implements the /_ah/remote_api endpoint.
This endpoint is used by offline tools such as the bulk loader.
*/
package remote_api // import "google.golang.org/appengine/remote_api"

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strconv"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal"
	pb "google.golang.org/appengine/internal/remote_api"
	"google.golang.org/appengine/log"
	"google.golang.org/appengine/user"
)

func init() {
	http.HandleFunc("/_ah/remote_api", handle)
}

func handle(w http.ResponseWriter, req *http.Request) {
	c := appengine.NewContext(req)

	u := user.Current(c)
	if u == nil {
		u, _ = user.CurrentOAuth(c,
			"https://www.googleapis.com/auth/cloud-platform",
			"https://www.googleapis.com/auth/appengine.apis",
		)
	}

	if u == nil || !u.Admin {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusUnauthorized)
		io.WriteString(w, "You must be logged in as an administrator to access this.\n")
		return
	}
	if req.Header.Get("X-Appcfg-Api-Version") == "" {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusForbidden)
		io.WriteString(w, "This request did not contain a necessary header.\n")
		return
	}

	if req.Method != "POST" {
		// Response must be YAML.
		rtok := req.FormValue("rtok")
		if rtok == "" {
			rtok = "0"
		}
		w.Header().Set("Content-Type", "text/yaml; charset=utf-8")
		fmt.Fprintf(w, `{app_id: %q, rtok: %q}`, internal.FullyQualifiedAppID(c), rtok)
		return
	}

	defer req.Body.Close()
	body, err := ioutil.ReadAll(req.Body)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		log.Errorf(c, "Failed reading body: %v", err)
		return
	}
	remReq := &pb.Request{}
	if err := proto.Unmarshal(body, remReq); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		log.Errorf(c, "Bad body: %v", err)
		return
	}

	service, method := *remReq.ServiceName, *remReq.Method
	if !requestSupported(service, method) {
		w.WriteHeader(http.StatusBadRequest)
		log.Errorf(c, "Unsupported RPC /%s.%s", service, method)
		return
	}

	rawReq := &rawMessage{remReq.Request}
	rawRes := &rawMessage{}
	err = internal.Call(c, service, method, rawReq, rawRes)

	remRes := &pb.Response{}
	if err == nil {
		remRes.Response = rawRes.buf
	} else if ae, ok := err.(*internal.APIError); ok {
		remRes.ApplicationError = &pb.ApplicationError{
			Code:   &ae.Code,
			Detail: &ae.Detail,
		}
	} else {
		// This shouldn't normally happen.
		log.Errorf(c, "appengine/remote_api: Unexpected error of type %T: %v", err, err)
		remRes.ApplicationError = &pb.ApplicationError{
			Code:   proto.Int32(0),
			Detail: proto.String(err.Error()),
		}
	}
	out, err := proto.Marshal(remRes)
	if err != nil {
		// This should not be possible.
		w.WriteHeader(500)
		log.Errorf(c, "proto.Marshal: %v", err)
		return
	}

	log.Infof(c, "Spooling %d bytes of response to /%s.%s", len(out), service, method)
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", strconv.Itoa(len(out)))
	w.Write(out)
}

// rawMessage is a protocol buffer type that is already serialised.
// This allows the remote_api code here to handle messages
// without having to know the real type.
type rawMessage struct {
	buf []byte
}

func (rm *rawMessage) Marshal() ([]byte, error) {
	return rm.buf, nil
}

func (rm *rawMessage) Unmarshal(buf []byte) error {
	rm.buf = make([]byte, len(buf))
	copy(rm.buf, buf)
	return nil
}

func requestSupported(service, method string) bool {
	// This list of supported services is taken from SERVICE_PB_MAP in remote_api_services.py
	switch service {
	case "app_identity_service", "blobstore", "capability_service", "channel", "datastore_v3",
		"datastore_v4", "file", "images", "logservice", "mail", "matcher", "memcache", "remote_datastore",
		"remote_socket", "search", "modules", "system", "taskqueue", "urlfetch", "user", "xmpp":
		return true
	}
	return false
}

// Methods to satisfy proto.Message.
func (rm *rawMessage) Reset()         { rm.buf = nil }
func (rm *rawMessage) String() string { return strconv.Quote(string(rm.buf)) }
func (*rawMessage) ProtoMessage()     {}
