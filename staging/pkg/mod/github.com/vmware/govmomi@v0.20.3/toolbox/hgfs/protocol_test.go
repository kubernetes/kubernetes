/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package hgfs

import (
	"bytes"
	"encoding/base64"
	"io"
	"testing"
)

func TestProtocolEncoding(t *testing.T) {
	ps := packetSize
	defer func() {
		packetSize = ps
	}()

	// a few structs have pading of some sort, leave PacketSize as-is for now with these tests
	packetSize = func(r *Packet) uint32 {
		return r.PacketSize
	}

	decode := func(s string) []byte {
		b, _ := base64.StdEncoding.DecodeString(s)
		return b
	}

	// base64 encoded packets below were captured from vmtoolsd during:
	// govc guest.download /etc/hosts -
	// govc guest.upload /etc/hosts /tmp/hosts
	tests := []struct {
		pkt string
		dec interface{}
	}{
		{
			"AQAAAP8AAABMAAAANAAAAAAAAIApAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA+AAAAAAAAAAAAAAAAAAAAAAAAA==",
			new(RequestCreateSessionV4),
		},
		{
			"AQAAAP8AAABYAgAANAAAAAAAAIApAAAAAAAAAAIAAAAAAAAA//////////8AAAAAAAAAAF/NcjwZ1DYAQQAAAAD4AAAAAAAAAQAAAAAAAAAAAAAAAQAAAAEAAAABAAAAAgAAAAEAAAADAAAAAQAAAAQAAAABAAAABQAAAAEAAAAGAAAAAQAAAAcAAAABAAAACAAAAAEAAAAJAAAAAQAAAAoAAAABAAAACwAAAAEAAAAMAAAAAQAAAA0AAAABAAAADgAAAAEAAAAPAAAAAQAAABAAAAABAAAAEQAAAAEAAAASAAAAAQAAABMAAAAAAAAAFAAAAAEAAAAVAAAAAQAAABYAAAABAAAAFwAAAAEAAAAYAAAAAQAAABkAAAABAAAAGgAAAAEAAAAbAAAAAQAAABwAAAABAAAAHQAAAAEAAAAeAAAAAQAAAB8AAAABAAAAIAAAAAEAAAAhAAAAAQAAACIAAAABAAAAIwAAAAEAAAAkAAAAAQAAACUAAAABAAAAJgAAAAEAAAAnAAAAAAAAACgAAAAAAAAAKQAAAAEAAAAqAAAAAQAAACsAAAAAAAAALAAAAAAAAAAtAAAAAAAAAC4AAAAAAAAALwAAAAAAAAAwAAAAAAAAADEAAAAAAAAAMgAAAAAAAAAzAAAAAAAAADQAAAAAAAAANQAAAAAAAAA2AAAAAAAAADcAAAAAAAAAOAAAAAAAAAA5AAAAAAAAADoAAAAAAAAAOwAAAAAAAAA8AAAAAAAAAD0AAAAAAAAAPgAAAAAAAAA/AAAAAAAAAEAAAAAAAAAA",
			new(ReplyCreateSessionV4),
		},
		{
			"AQAAAP8AAABbAAAANAAAAAAAAIAPAAAAAAAAAAEAAAAAAAAAX81yPBnUNgAAAAAAAAAAAAAAAIAPAAAAAAAAAAAAAAAAAACADgAAAHJvb3QAZXRjAGhvc3RzAA==",
			new(RequestGetattrV2),
		},
		{
			"AQAAAP8AAACpAAAANAAAAAAAAIAPAAAAAAAAAAIAAAAAAAAAX81yPBnUNgAAAAAAAAAAAAAAAAAAAAAA//sCAAAAAAAAAAAAxgAAAAAAAABkP/0QmUzSAaAHo3qfz9IBZD/9EJlM0gFkP/0QmUzSAQAGBAQAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAABY9xMAAAAAAAD8AAAEAAAAAAAAAAAAAAAAAAAAAA==",
			new(ReplyGetattrV2),
		},
		{
			"AQAAAP8AAABYAAAANAAAAAAAAIAAAAAAAAAAAAEAAAAAAAAAX81yPBnUNgAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAGDgAAAHJvb3QAZXRjAGhvc3Rzcw==",
			new(RequestOpen),
		},
		{
			"AQAAAP8AAABAAAAANAAAAAAAAIAAAAAAAAAAAAIAAAAAAAAAX81yPBnUNgAAAAAAAAAAAAAAAAAAAAAAAAAAAA==",
			new(ReplyOpen),
		},
		{
			"AQAAAP8AAABMAAAANAAAAAAAAIAZAAAAAAAAAAEAAAAAAAAAX81yPBnUNgAAAAAAAAAAAAAAAAAAAAAAAAAAAADwAAAAAAAAAAAAAA==",
			new(RequestReadV3),
		},
		{
			"AQAAAP8AAAAHAQAANAAAAAAAAIAZAAAAAAAAAAIAAAAAAAAAX81yPBnUNgAAAAAAAAAAAMYAAAAAAAAAAAAAADEyNy4wLjAuMQlsb2NhbGhvc3QKMTI3LjAuMS4xCXZhZ3JhbnQudm0JdmFncmFudAoKIyBUaGUgZm9sbG93aW5nIGxpbmVzIGFyZSBkZXNpcmFibGUgZm9yIElQdjYgY2FwYWJsZSBob3N0cwo6OjEgICAgIGxvY2FsaG9zdCBpcDYtbG9jYWxob3N0IGlwNi1sb29wYmFjawpmZjAyOjoxIGlwNi1hbGxub2RlcwpmZjAyOjoyIGlwNi1hbGxyb3V0ZXJzCgA=",
			new(ReplyReadV3),
		},
		{
			"AQAAAP8AAABAAAAANAAAAAAAAIADAAAAAAAAAAEAAAAAAAAAX81yPBnUNgAAAAAAAAAAAAAAAIADAAAAAAAAAA==",
			new(RequestClose),
		},
		{
			"AQAAAP8AAAA8AAAANAAAAAAAAIADAAAAAAAAAAIAAAAAAAAAX81yPBnUNgAAAAAAAAAAAAAAAAAAAAAA",
			new(ReplyClose),
		},
		{
			"AQAAAP8AAAA8AAAANAAAAAAAAIAqAAAAAAAAAAEAAAAAAAAAX81yPBnUNgAAAAAAAAAAAAAAAAAAAAAA",
			new(RequestDestroySessionV4),
		},
		{
			"AQAAAP8AAAA8AAAANAAAAAAAAIAqAAAAAAAAAAIAAAAAAAAAX81yPBnUNgAAAAAAAAAAAAAAAAAAAAAA",
			new(ReplyDestroySessionV4),
		},
		{
			"AQAAAP8AAACZAAAANAAAAAAAAIAYAAAAAAAAAAEAAAAAAAAAyXxnMKrzOwAAAAAAAAAAAAsIAAAAAAAAAQAAAAQAAAAABgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUAAAAAAAAAAAAAAAAAAAAcm9vdAB0bXAAcmVzb2x2LmNvbmYA",
			new(RequestOpenV3),
		},
		{
			"AQAAAP8AAABEAAAANAAAAAAAAIAYAAAAAAAAAAIAAAAAAAAAyXxnMKrzOwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
			new(ReplyOpenV3),
		},
		{
			"AQAAAP8AAADJAAAANAAAAAAAAIAQAAAAAAAAAAEAAAAAAAAAyXxnMKrzOwAAAAAAAAAAAAAAAIAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUAAAAcm9vdAB0bXAAcmVzb2x2LmNvbmYA",
			new(RequestSetattrV2),
		},
		{
			"AQAAAP8AAAA8AAAANAAAAAAAAIAQAAAAAAAAAAIAAAAAAAAAyXxnMKrzOwAAAAAAAAAAAAAAAAAAAAAA",
			new(ReplySetattrV2),
		},
		{
			"AQAAAP8AAACTAAAANAAAAAAAAIAaAAAAAAAAAAEAAAAAAAAAyXxnMKrzOwAAAAAAAAAAAAAAAAABAAAAAAAAAABGAAAAAAAAAAAAAABuYW1lc2VydmVyIDEwLjExOC42NS4xCm5hbWVzZXJ2ZXIgMTAuMTE4LjY1LjIKc2VhcmNoIGVuZy52bXdhcmUuY29tIAoK",
			new(RequestWriteV3),
		},
		{
			"AQAAAP8AAABAAAAANAAAAAAAAIAaAAAAAAAAAAIAAAAAAAAAyXxnMKrzOwAAAAAAAAAAAEYAAAAAAAAAAAAAAA==",
			new(ReplyWriteV3),
		},
	}

	for i, test := range tests {
		dec := decode(test.pkt)
		pkt := new(Packet)
		err := pkt.UnmarshalBinary(dec)
		if err != nil {
			t.Fatal(err)
		}

		err = UnmarshalBinary(pkt.Payload, test.dec)
		if err != nil {
			t.Fatal(err)
		}

		pkt.Payload, err = MarshalBinary(test.dec)
		if err != nil {
			t.Fatal(err)
		}

		enc, err := pkt.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.HasPrefix(dec, enc) {
			t.Errorf("%d: %T != %s\n", i, test.dec, test.pkt)
		}
	}
}

func TestFileName(t *testing.T) {
	tests := []struct {
		raw  string
		name string
	}{
		{
			"root\x00etc\x00hosts",
			"/etc/hosts",
		},
	}

	for i, test := range tests {
		fn := FileName{
			Name:   test.raw,
			Length: uint32(len(test.raw)),
		}

		if fn.Path() != test.name {
			t.Errorf("%d: %q != %q", i, fn.Path(), test.name)
		}

		var fs FileName
		fs.FromString(test.name)
		if fs != fn {
			t.Errorf("%d: %v != %v", i, fn, fs)
		}
	}
}

func TestProtocolMarshal(t *testing.T) {
	var buf []byte
	var x uint64
	err := UnmarshalBinary(buf, x)
	if err == nil {
		t.Fatal("expected error")
	}

	status := err.(*Status)
	if status.Error() != io.EOF.Error() {
		t.Errorf("err=%s", status.Err)
	}

	status.Err = nil
	if status.Error() == "" {
		t.Error("status")
	}
}
