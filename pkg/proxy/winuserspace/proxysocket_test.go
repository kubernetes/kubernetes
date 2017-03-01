/*
Copyright 2017 The Kubernetes Authors.

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

package winuserspace

import (
	"reflect"
	"testing"
)

func TestPackUnpackDnsMsgUnqualifiedName(t *testing.T) {
	msg := &dnsMsg{}
	var buffer [4096]byte

	msg.header.id = 1
	msg.header.qdCount = 1
	msg.question = make([]dnsQuestion, msg.header.qdCount)
	msg.question[0].qClass = 0x01
	msg.question[0].qType = 0x01
	msg.question[0].qName.name = "kubernetes"

	length, ok := msg.packDnsMsg(buffer[:])
	if !ok {
		t.Errorf("Pack DNS message failed.")
	}

	unpackedMsg := &dnsMsg{}
	if !unpackedMsg.unpackDnsMsg(buffer[:length]) {
		t.Errorf("Unpack DNS message failed.")
	}

	if !reflect.DeepEqual(msg, unpackedMsg) {
		t.Errorf("Pack and Unpack DNS message are not consistent.")
	}
}

func TestPackUnpackDnsMsgFqdn(t *testing.T) {
	msg := &dnsMsg{}
	var buffer [4096]byte

	msg.header.id = 1
	msg.header.qdCount = 1
	msg.question = make([]dnsQuestion, msg.header.qdCount)
	msg.question[0].qClass = 0x01
	msg.question[0].qType = 0x01
	msg.question[0].qName.name = "kubernetes.default.svc.cluster.local"

	length, ok := msg.packDnsMsg(buffer[:])
	if !ok {
		t.Errorf("Pack DNS message failed.")
	}

	unpackedMsg := &dnsMsg{}
	if !unpackedMsg.unpackDnsMsg(buffer[:length]) {
		t.Errorf("Unpack DNS message failed.")
	}

	if !reflect.DeepEqual(msg, unpackedMsg) {
		t.Errorf("Pack and Unpack DNS message are not consistent.")
	}
}

func TestPackUnpackDnsMsgEmptyName(t *testing.T) {
	msg := &dnsMsg{}
	var buffer [4096]byte

	msg.header.id = 1
	msg.header.qdCount = 1
	msg.question = make([]dnsQuestion, msg.header.qdCount)
	msg.question[0].qClass = 0x01
	msg.question[0].qType = 0x01
	msg.question[0].qName.name = ""

	length, ok := msg.packDnsMsg(buffer[:])
	if !ok {
		t.Errorf("Pack DNS message failed.")
	}

	unpackedMsg := &dnsMsg{}
	if !unpackedMsg.unpackDnsMsg(buffer[:length]) {
		t.Errorf("Unpack DNS message failed.")
	}

	if !reflect.DeepEqual(msg, unpackedMsg) {
		t.Errorf("Pack and Unpack DNS message are not consistent.")
	}
}

func TestPackUnpackDnsMsgMultipleQuestions(t *testing.T) {
	msg := &dnsMsg{}
	var buffer [4096]byte

	msg.header.id = 1
	msg.header.qdCount = 2
	msg.question = make([]dnsQuestion, msg.header.qdCount)
	msg.question[0].qClass = 0x01
	msg.question[0].qType = 0x01
	msg.question[0].qName.name = "kubernetes"
	msg.question[1].qClass = 0x01
	msg.question[1].qType = 0x1c
	msg.question[1].qName.name = "kubernetes.default"

	length, ok := msg.packDnsMsg(buffer[:])
	if !ok {
		t.Errorf("Pack DNS message failed.")
	}

	unpackedMsg := &dnsMsg{}
	if !unpackedMsg.unpackDnsMsg(buffer[:length]) {
		t.Errorf("Unpack DNS message failed.")
	}

	if !reflect.DeepEqual(msg, unpackedMsg) {
		t.Errorf("Pack and Unpack DNS message are not consistent.")
	}
}
