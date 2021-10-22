// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"strings"
	"testing"
)

func TestClientVersion(t *testing.T) {
	for _, tt := range []struct {
		name      string
		version   string
		multiLine string
		wantErr   bool
	}{
		{
			name:    "default version",
			version: packageVersion,
		},
		{
			name:    "custom version",
			version: "SSH-2.0-CustomClientVersionString",
		},
		{
			name:      "good multi line version",
			version:   packageVersion,
			multiLine: strings.Repeat("ignored\r\n", 20),
		},
		{
			name:      "bad multi line version",
			version:   packageVersion,
			multiLine: "bad multi line version",
			wantErr:   true,
		},
		{
			name:      "long multi line version",
			version:   packageVersion,
			multiLine: strings.Repeat("long multi line version\r\n", 50)[:256],
			wantErr:   true,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c1, c2, err := netPipe()
			if err != nil {
				t.Fatalf("netPipe: %v", err)
			}
			defer c1.Close()
			defer c2.Close()
			go func() {
				if tt.multiLine != "" {
					c1.Write([]byte(tt.multiLine))
				}
				NewClientConn(c1, "", &ClientConfig{
					ClientVersion:   tt.version,
					HostKeyCallback: InsecureIgnoreHostKey(),
				})
				c1.Close()
			}()
			conf := &ServerConfig{NoClientAuth: true}
			conf.AddHostKey(testSigners["rsa"])
			conn, _, _, err := NewServerConn(c2, conf)
			if err == nil == tt.wantErr {
				t.Fatalf("got err %v; wantErr %t", err, tt.wantErr)
			}
			if tt.wantErr {
				// Don't verify the version on an expected error.
				return
			}
			if got := string(conn.ClientVersion()); got != tt.version {
				t.Fatalf("got %q; want %q", got, tt.version)
			}
		})
	}
}

func TestHostKeyCheck(t *testing.T) {
	for _, tt := range []struct {
		name      string
		wantError string
		key       PublicKey
	}{
		{"no callback", "must specify HostKeyCallback", nil},
		{"correct key", "", testSigners["rsa"].PublicKey()},
		{"mismatch", "mismatch", testSigners["ecdsa"].PublicKey()},
	} {
		c1, c2, err := netPipe()
		if err != nil {
			t.Fatalf("netPipe: %v", err)
		}
		defer c1.Close()
		defer c2.Close()
		serverConf := &ServerConfig{
			NoClientAuth: true,
		}
		serverConf.AddHostKey(testSigners["rsa"])

		go NewServerConn(c1, serverConf)
		clientConf := ClientConfig{
			User: "user",
		}
		if tt.key != nil {
			clientConf.HostKeyCallback = FixedHostKey(tt.key)
		}

		_, _, _, err = NewClientConn(c2, "", &clientConf)
		if err != nil {
			if tt.wantError == "" || !strings.Contains(err.Error(), tt.wantError) {
				t.Errorf("%s: got error %q, missing %q", tt.name, err.Error(), tt.wantError)
			}
		} else if tt.wantError != "" {
			t.Errorf("%s: succeeded, but want error string %q", tt.name, tt.wantError)
		}
	}
}

func TestBannerCallback(t *testing.T) {
	c1, c2, err := netPipe()
	if err != nil {
		t.Fatalf("netPipe: %v", err)
	}
	defer c1.Close()
	defer c2.Close()

	serverConf := &ServerConfig{
		PasswordCallback: func(conn ConnMetadata, password []byte) (*Permissions, error) {
			return &Permissions{}, nil
		},
		BannerCallback: func(conn ConnMetadata) string {
			return "Hello World"
		},
	}
	serverConf.AddHostKey(testSigners["rsa"])
	go NewServerConn(c1, serverConf)

	var receivedBanner string
	var bannerCount int
	clientConf := ClientConfig{
		Auth: []AuthMethod{
			Password("123"),
		},
		User:            "user",
		HostKeyCallback: InsecureIgnoreHostKey(),
		BannerCallback: func(message string) error {
			bannerCount++
			receivedBanner = message
			return nil
		},
	}

	_, _, _, err = NewClientConn(c2, "", &clientConf)
	if err != nil {
		t.Fatal(err)
	}

	if bannerCount != 1 {
		t.Errorf("got %d banners; want 1", bannerCount)
	}

	expected := "Hello World"
	if receivedBanner != expected {
		t.Fatalf("got %s; want %s", receivedBanner, expected)
	}
}
