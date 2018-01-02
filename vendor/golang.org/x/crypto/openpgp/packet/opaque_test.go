// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packet

import (
	"bytes"
	"encoding/hex"
	"io"
	"testing"
)

// Test packet.Read error handling in OpaquePacket.Parse,
// which attempts to re-read an OpaquePacket as a supported
// Packet type.
func TestOpaqueParseReason(t *testing.T) {
	buf, err := hex.DecodeString(UnsupportedKeyHex)
	if err != nil {
		t.Fatal(err)
	}
	or := NewOpaqueReader(bytes.NewBuffer(buf))
	count := 0
	badPackets := 0
	var uid *UserId
	for {
		op, err := or.Next()
		if err == io.EOF {
			break
		} else if err != nil {
			t.Errorf("#%d: opaque read error: %v", count, err)
			break
		}
		// try to parse opaque packet
		p, err := op.Parse()
		switch pkt := p.(type) {
		case *UserId:
			uid = pkt
		case *OpaquePacket:
			// If an OpaquePacket can't re-parse, packet.Read
			// certainly had its reasons.
			if pkt.Reason == nil {
				t.Errorf("#%d: opaque packet, no reason", count)
			} else {
				badPackets++
			}
		}
		count++
	}

	const expectedBad = 3
	// Test post-conditions, make sure we actually parsed packets as expected.
	if badPackets != expectedBad {
		t.Errorf("unexpected # unparseable packets: %d (want %d)", badPackets, expectedBad)
	}
	if uid == nil {
		t.Errorf("failed to find expected UID in unsupported keyring")
	} else if uid.Id != "Armin M. Warda <warda@nephilim.ruhr.de>" {
		t.Errorf("unexpected UID: %v", uid.Id)
	}
}

// This key material has public key and signature packet versions modified to
// an unsupported value (1), so that trying to parse the OpaquePacket to
// a typed packet will get an error. It also contains a GnuPG trust packet.
// (Created with: od -An -t x1 pubring.gpg | xargs | sed 's/ //g')
const UnsupportedKeyHex = `988d012e7a18a20000010400d6ac00d92b89c1f4396c243abb9b76d2e9673ad63483291fed88e22b82e255e441c078c6abbbf7d2d195e50b62eeaa915b85b0ec20c225ce2c64c167cacb6e711daf2e45da4a8356a059b8160e3b3628ac0dd8437b31f06d53d6e8ea4214d4a26406a6b63e1001406ef23e0bb3069fac9a99a91f77dfafd5de0f188a5da5e3c9000511b42741726d696e204d2e205761726461203c7761726461406e657068696c696d2e727568722e64653e8900950105102e8936c705d1eb399e58489901013f0e03ff5a0c4f421e34fcfa388129166420c08cd76987bcdec6f01bd0271459a85cc22048820dd4e44ac2c7d23908d540f54facf1b36b0d9c20488781ce9dca856531e76e2e846826e9951338020a03a09b57aa5faa82e9267458bd76105399885ac35af7dc1cbb6aaed7c39e1039f3b5beda2c0e916bd38560509bab81235d1a0ead83b0020000`
