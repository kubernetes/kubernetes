/*
Copyright (c) 2019 VMware, Inc. All Rights Reserved.

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
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type HttpNfcLease struct {
	mo.HttpNfcLease
	files map[string]string
}

var (
	nfcLease  sync.Map // HTTP access to NFC leases are token based and do not require Session auth
	nfcPrefix = "/nfc/"
)

// ServeNFC handles NFC file upload/download
func ServeNFC(w http.ResponseWriter, r *http.Request) {
	p := strings.Split(r.URL.Path, "/")
	id, name := p[len(p)-2], p[len(p)-1]
	ref := types.ManagedObjectReference{Type: "HttpNfcLease", Value: id}
	l, ok := nfcLease.Load(ref)
	if !ok {
		log.Printf("invalid NFC lease: %s", id)
		http.NotFound(w, r)
		return
	}
	lease := l.(*HttpNfcLease)
	file, ok := lease.files[name]
	if !ok {
		log.Printf("invalid NFC device id: %s", name)
		http.NotFound(w, r)
		return
	}

	status := http.StatusOK
	var dst io.Writer
	var src io.ReadCloser

	switch r.Method {
	case http.MethodPut, http.MethodPost:
		dst = ioutil.Discard
		src = r.Body
	case http.MethodGet:
		f, err := os.Open(file)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		src = f
	default:
		status = http.StatusMethodNotAllowed
	}

	n, err := io.Copy(dst, src)
	_ = src.Close()

	msg := fmt.Sprintf("transferred %d bytes", n)
	if err != nil {
		status = http.StatusInternalServerError
		msg = err.Error()
	}
	tracef("nfc %s %s: %s", r.Method, file, msg)
	w.WriteHeader(status)
}

func NewHttpNfcLease(ctx *Context, entity types.ManagedObjectReference) *HttpNfcLease {
	lease := &HttpNfcLease{
		HttpNfcLease: mo.HttpNfcLease{
			Info: &types.HttpNfcLeaseInfo{
				Entity:       entity,
				LeaseTimeout: 30000,
			},
			State: types.HttpNfcLeaseStateReady,
		},
		files: make(map[string]string),
	}

	ctx.Session.Put(lease)
	nfcLease.Store(lease.Reference(), lease)

	return lease
}

func (l *HttpNfcLease) HttpNfcLeaseComplete(ctx *Context, req *types.HttpNfcLeaseComplete) soap.HasFault {
	ctx.Session.Remove(ctx, req.This)
	nfcLease.Delete(req.This)

	return &methods.HttpNfcLeaseCompleteBody{
		Res: new(types.HttpNfcLeaseCompleteResponse),
	}
}

func (l *HttpNfcLease) HttpNfcLeaseAbort(ctx *Context, req *types.HttpNfcLeaseAbort) soap.HasFault {
	ctx.Session.Remove(ctx, req.This)
	nfcLease.Delete(req.This)

	return &methods.HttpNfcLeaseAbortBody{
		Res: new(types.HttpNfcLeaseAbortResponse),
	}
}

func (l *HttpNfcLease) HttpNfcLeaseProgress(ctx *Context, req *types.HttpNfcLeaseProgress) soap.HasFault {
	l.TransferProgress = req.Percent

	return &methods.HttpNfcLeaseProgressBody{
		Res: new(types.HttpNfcLeaseProgressResponse),
	}
}
