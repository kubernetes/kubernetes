// +build windows
// Copyright 2016 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package storageos

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"

	"github.com/Microsoft/go-winio"
)

const (
	nativeProtocol     = namedPipeProtocol
	nativeRealEndpoint = "npipe:////./pipe/docker_engine"
	nativeBadEndpoint  = "npipe:////./pipe/godockerclient_test_echo"
)

var (
	// namedPipeCount is used to provide uniqueness in the named pipes generated
	// by newNativeServer
	namedPipeCount int
	// namedPipesAllocateLock protects namedPipeCount
	namedPipeAllocateLock sync.Mutex
)

func newNativeServer(handler http.Handler) (*httptest.Server, func(), error) {
	namedPipeAllocateLock.Lock()
	defer namedPipeAllocateLock.Unlock()
	pipeName := fmt.Sprintf("//./pipe/godockerclient_test_%d", namedPipeCount)
	namedPipeCount++
	l, err := winio.ListenPipe(pipeName, &winio.PipeConfig{MessageMode: true})
	if err != nil {
		return nil, nil, err
	}
	srv := httptest.NewUnstartedServer(handler)
	srv.Listener = l
	return srv, func() {}, nil
}
