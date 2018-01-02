// Copyright 2015 Google Inc. All Rights Reserved.
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

package flags

import (
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUriString(t *testing.T) {
	tests := [...]struct {
		in   Uri
		want string
	}{
		{
			Uri{
				Key: "gcm",
			},
			"gcm",
		},
		{
			Uri{
				Key: "influxdb",
				Val: url.URL{
					Scheme:   "http",
					Host:     "monitoring-influxdb:8086",
					RawQuery: "key=val&key2=val2",
				},
			},
			"influxdb:http://monitoring-influxdb:8086?key=val&key2=val2",
		},
	}
	for _, c := range tests {
		assert.Equal(t, c.want, c.in.String())
	}
}

func TestUriSet(t *testing.T) {
	tests := [...]struct {
		in      string
		want    Uri
		wantErr bool
	}{
		{"", Uri{}, true},
		{":", Uri{}, true},
		{":foo", Uri{}, true},
		{"key:incorrecturl/%gh&%ij", Uri{}, true},
		{
			"gcm",
			Uri{Key: "gcm"},
			false,
		},
		{
			"gcm:",
			Uri{Key: "gcm"},
			false,
		},
		{
			"influxdb:http://monitoring-influxdb:8086?key=val&key2=val2",
			Uri{
				Key: "influxdb",
				Val: url.URL{
					Scheme:   "http",
					Host:     "monitoring-influxdb:8086",
					RawQuery: "key=val&key2=val2",
				},
			},
			false,
		},
		{
			"gcm:?metrics=all",
			Uri{
				Key: "gcm",
				Val: url.URL{
					RawQuery: "metrics=all",
				},
			},
			false,
		},
	}
	for _, c := range tests {
		var uri Uri
		err := uri.Set(c.in)
		if c.wantErr {
			assert.NotNil(t, err)
		} else {
			assert.Nil(t, err)
			assert.Equal(t, c.want, uri)
		}
	}
}

func TestUrisString(t *testing.T) {
	tests := [...]struct {
		in   Uris
		want string
	}{
		{
			Uris{
				Uri{Key: "gcm"},
			},
			"[gcm]",
		},
		{
			Uris{
				Uri{Key: "gcm"},
				Uri{
					Key: "influxdb",
					Val: url.URL{Path: "foo"},
				},
			},
			"[gcm influxdb:foo]",
		},
	}
	for _, c := range tests {
		assert.Equal(t, c.want, c.in.String())
	}
}

func TestUrisSet(t *testing.T) {
	tests := [...]struct {
		in      []string
		want    Uris
		wantErr bool
	}{
		{[]string{""}, Uris{}, true},
		{[]string{":foo"}, Uris{}, true},
		{
			[]string{"gcm"},
			Uris{
				Uri{Key: "gcm"},
			},
			false,
		},
		{
			[]string{"gcm", "influxdb:foo"},
			Uris{
				Uri{Key: "gcm"},
				Uri{
					Key: "influxdb",
					Val: url.URL{Path: "foo"},
				},
			},
			false,
		},
	}
	for _, c := range tests {
		var uris Uris
		var err error
		for _, s := range c.in {
			if err = uris.Set(s); err != nil {
				break
			}
		}
		if c.wantErr {
			assert.NotNil(t, err)
		} else {
			assert.Nil(t, err)
			assert.Equal(t, c.want, uris)
		}
	}
}
