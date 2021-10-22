// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"reflect"
	"testing"
)

func TestFindAgreedAlgorithms(t *testing.T) {
	initKex := func(k *kexInitMsg) {
		if k.KexAlgos == nil {
			k.KexAlgos = []string{"kex1"}
		}
		if k.ServerHostKeyAlgos == nil {
			k.ServerHostKeyAlgos = []string{"hostkey1"}
		}
		if k.CiphersClientServer == nil {
			k.CiphersClientServer = []string{"cipher1"}

		}
		if k.CiphersServerClient == nil {
			k.CiphersServerClient = []string{"cipher1"}

		}
		if k.MACsClientServer == nil {
			k.MACsClientServer = []string{"mac1"}

		}
		if k.MACsServerClient == nil {
			k.MACsServerClient = []string{"mac1"}

		}
		if k.CompressionClientServer == nil {
			k.CompressionClientServer = []string{"compression1"}

		}
		if k.CompressionServerClient == nil {
			k.CompressionServerClient = []string{"compression1"}

		}
		if k.LanguagesClientServer == nil {
			k.LanguagesClientServer = []string{"language1"}

		}
		if k.LanguagesServerClient == nil {
			k.LanguagesServerClient = []string{"language1"}

		}
	}

	initDirAlgs := func(a *directionAlgorithms) {
		if a.Cipher == "" {
			a.Cipher = "cipher1"
		}
		if a.MAC == "" {
			a.MAC = "mac1"
		}
		if a.Compression == "" {
			a.Compression = "compression1"
		}
	}

	initAlgs := func(a *algorithms) {
		if a.kex == "" {
			a.kex = "kex1"
		}
		if a.hostKey == "" {
			a.hostKey = "hostkey1"
		}
		initDirAlgs(&a.r)
		initDirAlgs(&a.w)
	}

	type testcase struct {
		name                   string
		clientIn, serverIn     kexInitMsg
		wantClient, wantServer algorithms
		wantErr                bool
	}

	cases := []testcase{
		testcase{
			name: "standard",
		},

		testcase{
			name: "no common hostkey",
			serverIn: kexInitMsg{
				ServerHostKeyAlgos: []string{"hostkey2"},
			},
			wantErr: true,
		},

		testcase{
			name: "no common kex",
			serverIn: kexInitMsg{
				KexAlgos: []string{"kex2"},
			},
			wantErr: true,
		},

		testcase{
			name: "no common cipher",
			serverIn: kexInitMsg{
				CiphersClientServer: []string{"cipher2"},
			},
			wantErr: true,
		},

		testcase{
			name: "client decides cipher",
			serverIn: kexInitMsg{
				CiphersClientServer: []string{"cipher1", "cipher2"},
				CiphersServerClient: []string{"cipher2", "cipher3"},
			},
			clientIn: kexInitMsg{
				CiphersClientServer: []string{"cipher2", "cipher1"},
				CiphersServerClient: []string{"cipher3", "cipher2"},
			},
			wantClient: algorithms{
				r: directionAlgorithms{
					Cipher: "cipher3",
				},
				w: directionAlgorithms{
					Cipher: "cipher2",
				},
			},
			wantServer: algorithms{
				w: directionAlgorithms{
					Cipher: "cipher3",
				},
				r: directionAlgorithms{
					Cipher: "cipher2",
				},
			},
		},

		// TODO(hanwen): fix and add tests for AEAD ignoring
		// the MACs field
	}

	for i := range cases {
		initKex(&cases[i].clientIn)
		initKex(&cases[i].serverIn)
		initAlgs(&cases[i].wantClient)
		initAlgs(&cases[i].wantServer)
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			serverAlgs, serverErr := findAgreedAlgorithms(false, &c.clientIn, &c.serverIn)
			clientAlgs, clientErr := findAgreedAlgorithms(true, &c.clientIn, &c.serverIn)

			serverHasErr := serverErr != nil
			clientHasErr := clientErr != nil
			if c.wantErr != serverHasErr || c.wantErr != clientHasErr {
				t.Fatalf("got client/server error (%v, %v), want hasError %v",
					clientErr, serverErr, c.wantErr)

			}
			if c.wantErr {
				return
			}

			if !reflect.DeepEqual(serverAlgs, &c.wantServer) {
				t.Errorf("server: got algs %#v, want %#v", serverAlgs, &c.wantServer)
			}
			if !reflect.DeepEqual(clientAlgs, &c.wantClient) {
				t.Errorf("server: got algs %#v, want %#v", clientAlgs, &c.wantClient)
			}
		})
	}
}
