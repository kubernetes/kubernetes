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

import "testing"

func TestCtlV3TxnInteractiveSuccess(t *testing.T) {
	testCtl(t, txnTestSuccess, withInteractive())
}
func TestCtlV3TxnInteractiveSuccessNoTLS(t *testing.T) {
	testCtl(t, txnTestSuccess, withInteractive(), withCfg(configNoTLS))
}
func TestCtlV3TxnInteractiveSuccessClientTLS(t *testing.T) {
	testCtl(t, txnTestSuccess, withInteractive(), withCfg(configClientTLS))
}
func TestCtlV3TxnInteractiveSuccessPeerTLS(t *testing.T) {
	testCtl(t, txnTestSuccess, withInteractive(), withCfg(configPeerTLS))
}
func TestCtlV3TxnInteractiveFail(t *testing.T) {
	testCtl(t, txnTestFail, withInteractive())
}

func txnTestSuccess(cx ctlCtx) {
	if err := ctlV3Put(cx, "key1", "value1", ""); err != nil {
		cx.t.Fatalf("txnTestSuccess ctlV3Put error (%v)", err)
	}
	if err := ctlV3Put(cx, "key2", "value2", ""); err != nil {
		cx.t.Fatalf("txnTestSuccess ctlV3Put error (%v)", err)
	}
	rqs := []txnRequests{
		{
			compare:  []string{`value("key1") != "value2"`, `value("key2") != "value1"`},
			ifSucess: []string{"get key1", "get key2"},
			results:  []string{"SUCCESS", "key1", "value1", "key2", "value2"},
		},
		{
			compare:  []string{`version("key1") = "1"`, `version("key2") = "1"`},
			ifSucess: []string{"get key1", "get key2", `put "key \"with\" space" "value \x23"`},
			ifFail:   []string{`put key1 "fail"`, `put key2 "fail"`},
			results:  []string{"SUCCESS", "key1", "value1", "key2", "value2"},
		},
		{
			compare:  []string{`version("key \"with\" space") = "1"`},
			ifSucess: []string{`get "key \"with\" space"`},
			results:  []string{"SUCCESS", `key "with" space`, "value \x23"},
		},
	}
	for _, rq := range rqs {
		if err := ctlV3Txn(cx, rq); err != nil {
			cx.t.Fatal(err)
		}
	}
}

func txnTestFail(cx ctlCtx) {
	if err := ctlV3Put(cx, "key1", "value1", ""); err != nil {
		cx.t.Fatalf("txnTestSuccess ctlV3Put error (%v)", err)
	}
	rqs := []txnRequests{
		{
			compare:  []string{`version("key") < "0"`},
			ifSucess: []string{`put key "success"`},
			ifFail:   []string{`put key "fail"`},
			results:  []string{"FAILURE", "OK"},
		},
		{
			compare:  []string{`value("key1") != "value1"`},
			ifSucess: []string{`put key1 "success"`},
			ifFail:   []string{`put key1 "fail"`},
			results:  []string{"FAILURE", "OK"},
		},
	}
	for _, rq := range rqs {
		if err := ctlV3Txn(cx, rq); err != nil {
			cx.t.Fatal(err)
		}
	}
}

type txnRequests struct {
	compare  []string
	ifSucess []string
	ifFail   []string
	results  []string
}

func ctlV3Txn(cx ctlCtx, rqs txnRequests) error {
	// TODO: support non-interactive mode
	cmdArgs := append(cx.PrefixArgs(), "txn")
	if cx.interactive {
		cmdArgs = append(cmdArgs, "--interactive")
	}
	proc, err := spawnCmd(cmdArgs)
	if err != nil {
		return err
	}
	_, err = proc.Expect("compares:")
	if err != nil {
		return err
	}
	for _, req := range rqs.compare {
		if err = proc.Send(req + "\r"); err != nil {
			return err
		}
	}
	if err = proc.Send("\r"); err != nil {
		return err
	}

	_, err = proc.Expect("success requests (get, put, delete):")
	if err != nil {
		return err
	}
	for _, req := range rqs.ifSucess {
		if err = proc.Send(req + "\r"); err != nil {
			return err
		}
	}
	if err = proc.Send("\r"); err != nil {
		return err
	}

	_, err = proc.Expect("failure requests (get, put, delete):")
	if err != nil {
		return err
	}
	for _, req := range rqs.ifFail {
		if err = proc.Send(req + "\r"); err != nil {
			return err
		}
	}
	if err = proc.Send("\r"); err != nil {
		return err
	}

	for _, line := range rqs.results {
		_, err = proc.Expect(line)
		if err != nil {
			return err
		}
	}
	return proc.Close()
}
