// +build x

// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

// These tests are used to verify msgpack and cbor implementations against their python libraries.
// If you have the library installed, you can enable the tests back by running: go test -tags=x .
// Look at test.py for how to setup your environment.

import (
	"testing"
)

func TestMsgpackPythonGenStreams(t *testing.T) {
	doTestPythonGenStreams(t, "msgpack", testMsgpackH)
}

func TestCborPythonGenStreams(t *testing.T) {
	doTestPythonGenStreams(t, "cbor", testCborH)
}

func TestMsgpackRpcSpecGoClientToPythonSvc(t *testing.T) {
	doTestMsgpackRpcSpecGoClientToPythonSvc(t)
}

func TestMsgpackRpcSpecPythonClientToGoSvc(t *testing.T) {
	doTestMsgpackRpcSpecPythonClientToGoSvc(t)
}
