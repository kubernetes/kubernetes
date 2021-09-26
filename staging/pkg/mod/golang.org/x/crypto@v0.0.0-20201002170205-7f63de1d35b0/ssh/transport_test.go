// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"strings"
	"testing"
)

func TestReadVersion(t *testing.T) {
	longVersion := strings.Repeat("SSH-2.0-bla", 50)[:253]
	multiLineVersion := strings.Repeat("ignored\r\n", 20) + "SSH-2.0-bla\r\n"
	cases := map[string]string{
		"SSH-2.0-bla\r\n":    "SSH-2.0-bla",
		"SSH-2.0-bla\n":      "SSH-2.0-bla",
		multiLineVersion:     "SSH-2.0-bla",
		longVersion + "\r\n": longVersion,
	}

	for in, want := range cases {
		result, err := readVersion(bytes.NewBufferString(in))
		if err != nil {
			t.Errorf("readVersion(%q): %s", in, err)
		}
		got := string(result)
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	}
}

func TestReadVersionError(t *testing.T) {
	longVersion := strings.Repeat("SSH-2.0-bla", 50)[:253]
	multiLineVersion := strings.Repeat("ignored\r\n", 50) + "SSH-2.0-bla\r\n"
	cases := []string{
		longVersion + "too-long\r\n",
		multiLineVersion,
	}
	for _, in := range cases {
		if _, err := readVersion(bytes.NewBufferString(in)); err == nil {
			t.Errorf("readVersion(%q) should have failed", in)
		}
	}
}

func TestExchangeVersionsBasic(t *testing.T) {
	v := "SSH-2.0-bla"
	buf := bytes.NewBufferString(v + "\r\n")
	them, err := exchangeVersions(buf, []byte("xyz"))
	if err != nil {
		t.Errorf("exchangeVersions: %v", err)
	}

	if want := "SSH-2.0-bla"; string(them) != want {
		t.Errorf("got %q want %q for our version", them, want)
	}
}

func TestExchangeVersions(t *testing.T) {
	cases := []string{
		"not\x000allowed",
		"not allowed\x01\r\n",
	}
	for _, c := range cases {
		buf := bytes.NewBufferString("SSH-2.0-bla\r\n")
		if _, err := exchangeVersions(buf, []byte(c)); err == nil {
			t.Errorf("exchangeVersions(%q): should have failed", c)
		}
	}
}

type closerBuffer struct {
	bytes.Buffer
}

func (b *closerBuffer) Close() error {
	return nil
}

func TestTransportMaxPacketWrite(t *testing.T) {
	buf := &closerBuffer{}
	tr := newTransport(buf, rand.Reader, true)
	huge := make([]byte, maxPacket+1)
	err := tr.writePacket(huge)
	if err == nil {
		t.Errorf("transport accepted write for a huge packet.")
	}
}

func TestTransportMaxPacketReader(t *testing.T) {
	var header [5]byte
	huge := make([]byte, maxPacket+128)
	binary.BigEndian.PutUint32(header[0:], uint32(len(huge)))
	// padding.
	header[4] = 0

	buf := &closerBuffer{}
	buf.Write(header[:])
	buf.Write(huge)

	tr := newTransport(buf, rand.Reader, true)
	_, err := tr.readPacket()
	if err == nil {
		t.Errorf("transport succeeded reading huge packet.")
	} else if !strings.Contains(err.Error(), "large") {
		t.Errorf("got %q, should mention %q", err.Error(), "large")
	}
}
