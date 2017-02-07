/*
Copyright 2016 The Kubernetes Authors.

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

package cert

import (
	"crypto/x509/pkix"
	"io/ioutil"
	"net"
	"testing"
)

func TestMakeCSR(t *testing.T) {
	keyFile := "testdata/dontUseThisKey.pem"
	subject := &pkix.Name{
		CommonName: "kube-worker",
	}
	dnsSANs := []string{"localhost"}
	ipSANs := []net.IP{net.ParseIP("127.0.0.1")}

	keyData, err := ioutil.ReadFile(keyFile)
	if err != nil {
		t.Fatal(err)
	}
	key, err := ParsePrivateKeyPEM(keyData)
	if err != nil {
		t.Fatal(err)
	}
	_, err = MakeCSR(key, subject, dnsSANs, ipSANs)
	if err != nil {
		t.Error(err)
	}
}
