// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package e2e

import (
	"encoding/json"
	"testing"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/pkg/testutil"
)

func TestV3CurlPutGetNoTLS(t *testing.T)     { testCurlPutGetGRPCGateway(t, &configNoTLS) }
func TestV3CurlPutGetAutoTLS(t *testing.T)   { testCurlPutGetGRPCGateway(t, &configAutoTLS) }
func TestV3CurlPutGetAllTLS(t *testing.T)    { testCurlPutGetGRPCGateway(t, &configTLS) }
func TestV3CurlPutGetPeerTLS(t *testing.T)   { testCurlPutGetGRPCGateway(t, &configPeerTLS) }
func TestV3CurlPutGetClientTLS(t *testing.T) { testCurlPutGetGRPCGateway(t, &configClientTLS) }
func testCurlPutGetGRPCGateway(t *testing.T, cfg *etcdProcessClusterConfig) {
	defer testutil.AfterTest(t)

	epc, err := newEtcdProcessCluster(cfg)
	if err != nil {
		t.Fatalf("could not start etcd process cluster (%v)", err)
	}
	defer func() {
		if cerr := epc.Close(); err != nil {
			t.Fatalf("error closing etcd processes (%v)", cerr)
		}
	}()

	var (
		key   = []byte("foo")
		value = []byte("bar") // this will be automatically base64-encoded by Go

		expectPut = `"revision":"`
		expectGet = `"value":"`
	)
	putData, err := json.Marshal(&pb.PutRequest{
		Key:   key,
		Value: value,
	})
	if err != nil {
		t.Fatal(err)
	}
	rangeData, err := json.Marshal(&pb.RangeRequest{
		Key: key,
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := cURLPost(epc, cURLReq{endpoint: "/v3alpha/kv/put", value: string(putData), expected: expectPut}); err != nil {
		t.Fatalf("failed put with curl (%v)", err)
	}
	if err := cURLPost(epc, cURLReq{endpoint: "/v3alpha/kv/range", value: string(rangeData), expected: expectGet}); err != nil {
		t.Fatalf("failed get with curl (%v)", err)
	}

	if cfg.clientTLS == clientTLSAndNonTLS {
		if err := cURLPost(epc, cURLReq{endpoint: "/v3alpha/kv/range", value: string(rangeData), expected: expectGet, isTLS: true}); err != nil {
			t.Fatalf("failed get with curl (%v)", err)
		}
	}
}

func TestV3CurlWatch(t *testing.T) {
	defer testutil.AfterTest(t)

	epc, err := newEtcdProcessCluster(&configNoTLS)
	if err != nil {
		t.Fatalf("could not start etcd process cluster (%v)", err)
	}
	defer func() {
		if cerr := epc.Close(); err != nil {
			t.Fatalf("error closing etcd processes (%v)", cerr)
		}
	}()

	// store "bar" into "foo"
	putreq, err := json.Marshal(&pb.PutRequest{Key: []byte("foo"), Value: []byte("bar")})
	if err != nil {
		t.Fatal(err)
	}
	if err = cURLPost(epc, cURLReq{endpoint: "/v3alpha/kv/put", value: string(putreq), expected: "revision"}); err != nil {
		t.Fatalf("failed put with curl (%v)", err)
	}
	// watch for first update to "foo"
	wcr := &pb.WatchCreateRequest{Key: []byte("foo"), StartRevision: 1}
	wreq, err := json.Marshal(wcr)
	if err != nil {
		t.Fatal(err)
	}
	// marshaling the grpc to json gives:
	// "{"RequestUnion":{"CreateRequest":{"key":"Zm9v","start_revision":1}}}"
	// but the gprc-gateway expects a different format..
	wstr := `{"create_request" : ` + string(wreq) + "}"
	// expects "bar", timeout after 2 seconds since stream waits forever
	if err = cURLPost(epc, cURLReq{endpoint: "/v3alpha/watch", value: wstr, expected: `"YmFy"`, timeout: 2}); err != nil {
		t.Fatal(err)
	}
}
