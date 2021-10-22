// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package strfmt

import (
	"strings"
	"testing"
	"time"

	"github.com/mitchellh/mapstructure"
	"github.com/stretchr/testify/assert"
)

type testFormat string

func (t testFormat) MarshalText() ([]byte, error) {
	return []byte(string(t)), nil
}

func (t *testFormat) UnmarshalText(b []byte) error {
	*t = testFormat(string(b))
	return nil
}

func (t testFormat) String() string {
	return string(t)
}

func isTestFormat(s string) bool {
	return strings.HasPrefix(s, "tf")
}

type tf2 string

func (t tf2) MarshalText() ([]byte, error) {
	return []byte(string(t)), nil
}

func (t *tf2) UnmarshalText(b []byte) error {
	*t = tf2(string(b))
	return nil
}

func istf2(s string) bool {
	return strings.HasPrefix(s, "af")
}

func (t tf2) String() string {
	return string(t)
}

type bf string

func (t bf) MarshalText() ([]byte, error) {
	return []byte(string(t)), nil
}

func (t *bf) UnmarshalText(b []byte) error {
	*t = bf(string(b))
	return nil
}

func (t bf) String() string {
	return string(t)
}

func isbf(s string) bool {
	return strings.HasPrefix(s, "bf")
}

func istf3(s string) bool {
	return strings.HasPrefix(s, "ff")
}

func init() {
	tf := testFormat("")
	Default.Add("test-format", &tf, isTestFormat)
}

func TestFormatRegistry(t *testing.T) {
	f2 := tf2("")
	f3 := bf("")
	registry := NewFormats()

	assert.True(t, registry.ContainsName("test-format"))
	assert.True(t, registry.ContainsName("testformat"))
	assert.False(t, registry.ContainsName("ttt"))

	assert.True(t, registry.Validates("testformat", "tfa"))
	assert.False(t, registry.Validates("testformat", "ffa"))

	assert.True(t, registry.Add("tf2", &f2, istf2))
	assert.True(t, registry.ContainsName("tf2"))
	assert.False(t, registry.ContainsName("tfw"))
	assert.True(t, registry.Validates("tf2", "afa"))

	assert.False(t, registry.Add("tf2", &f3, isbf))
	assert.True(t, registry.ContainsName("tf2"))
	assert.False(t, registry.ContainsName("tfw"))
	assert.True(t, registry.Validates("tf2", "bfa"))
	assert.False(t, registry.Validates("tf2", "afa"))

	assert.False(t, registry.Add("tf2", &f2, istf2))
	assert.True(t, registry.Add("tf3", &f2, istf3))
	assert.True(t, registry.ContainsName("tf3"))
	assert.True(t, registry.ContainsName("tf2"))
	assert.False(t, registry.ContainsName("tfw"))
	assert.True(t, registry.Validates("tf3", "ffa"))

	assert.True(t, registry.DelByName("tf3"))
	assert.True(t, registry.Add("tf3", &f2, istf3))

	assert.True(t, registry.DelByName("tf3"))
	assert.False(t, registry.DelByName("unknown"))
	assert.False(t, registry.Validates("unknown", ""))
}

type testStruct struct {
	D          Date       `json:"d,omitempty"`
	DT         DateTime   `json:"dt,omitempty"`
	Dur        Duration   `json:"dur,omitempty"`
	URI        URI        `json:"uri,omitempty"`
	Eml        Email      `json:"eml,omitempty"`
	UUID       UUID       `json:"uuid,omitempty"`
	UUID3      UUID3      `json:"uuid3,omitempty"`
	UUID4      UUID4      `json:"uuid4,omitempty"`
	UUID5      UUID5      `json:"uuid5,omitempty"`
	Hn         Hostname   `json:"hn,omitempty"`
	Ipv4       IPv4       `json:"ipv4,omitempty"`
	Ipv6       IPv6       `json:"ipv6,omitempty"`
	Cidr       CIDR       `json:"cidr,omitempty"`
	Mac        MAC        `json:"mac,omitempty"`
	Isbn       ISBN       `json:"isbn,omitempty"`
	Isbn10     ISBN10     `json:"isbn10,omitempty"`
	Isbn13     ISBN13     `json:"isbn13,omitempty"`
	Creditcard CreditCard `json:"creditcard,omitempty"`
	Ssn        SSN        `json:"ssn,omitempty"`
	Hexcolor   HexColor   `json:"hexcolor,omitempty"`
	Rgbcolor   RGBColor   `json:"rgbcolor,omitempty"`
	B64        Base64     `json:"b64,omitempty"`
	Pw         Password   `json:"pw,omitempty"`
}

func TestDecodeHook(t *testing.T) {
	registry := NewFormats()
	m := map[string]interface{}{
		"d":          "2014-12-15",
		"dt":         "2012-03-02T15:06:05.999999999Z",
		"dur":        "5s",
		"uri":        "http://www.dummy.com",
		"eml":        "dummy@dummy.com",
		"uuid":       "a8098c1a-f86e-11da-bd1a-00112444be1e",
		"uuid3":      "bcd02e22-68f0-3046-a512-327cca9def8f",
		"uuid4":      "025b0d74-00a2-4048-bf57-227c5111bb34",
		"uuid5":      "886313e1-3b8a-5372-9b90-0c9aee199e5d",
		"hn":         "somewhere.com",
		"ipv4":       "192.168.254.1",
		"ipv6":       "::1",
		"cidr":       "192.0.2.1/24",
		"mac":        "01:02:03:04:05:06",
		"isbn":       "0321751043",
		"isbn10":     "0321751043",
		"isbn13":     "978-0321751041",
		"hexcolor":   "#FFFFFF",
		"rgbcolor":   "rgb(255,255,255)",
		"pw":         "super secret stuff here",
		"ssn":        "111-11-1111",
		"creditcard": "4111-1111-1111-1111",
		"b64":        "ZWxpemFiZXRocG9zZXk=",
	}

	date, _ := time.Parse(RFC3339FullDate, "2014-12-15")
	dur, _ := ParseDuration("5s")
	dt, _ := ParseDateTime("2012-03-02T15:06:05.999999999Z")

	exp := &testStruct{
		D:          Date(date),
		DT:         dt,
		Dur:        Duration(dur),
		URI:        URI("http://www.dummy.com"),
		Eml:        Email("dummy@dummy.com"),
		UUID:       UUID("a8098c1a-f86e-11da-bd1a-00112444be1e"),
		UUID3:      UUID3("bcd02e22-68f0-3046-a512-327cca9def8f"),
		UUID4:      UUID4("025b0d74-00a2-4048-bf57-227c5111bb34"),
		UUID5:      UUID5("886313e1-3b8a-5372-9b90-0c9aee199e5d"),
		Hn:         Hostname("somewhere.com"),
		Ipv4:       IPv4("192.168.254.1"),
		Ipv6:       IPv6("::1"),
		Cidr:       CIDR("192.0.2.1/24"),
		Mac:        MAC("01:02:03:04:05:06"),
		Isbn:       ISBN("0321751043"),
		Isbn10:     ISBN10("0321751043"),
		Isbn13:     ISBN13("978-0321751041"),
		Creditcard: CreditCard("4111-1111-1111-1111"),
		Ssn:        SSN("111-11-1111"),
		Hexcolor:   HexColor("#FFFFFF"),
		Rgbcolor:   RGBColor("rgb(255,255,255)"),
		B64:        Base64("ZWxpemFiZXRocG9zZXk="),
		Pw:         Password("super secret stuff here"),
	}

	test := new(testStruct)
	cfg := &mapstructure.DecoderConfig{
		DecodeHook: registry.MapStructureHookFunc(),
		// weakly typed will pass if this passes
		WeaklyTypedInput: false,
		Result:           test,
	}
	d, err := mapstructure.NewDecoder(cfg)
	assert.Nil(t, err)
	err = d.Decode(m)
	assert.Nil(t, err)
	assert.Equal(t, exp, test)
}

func TestDecodeDateTimeHook(t *testing.T) {
	testCases := []struct {
		Name  string
		Input string
	}{
		{
			"empty datetime",
			"",
		},
		{
			"invalid non empty datetime",
			"2019-01-01",
		},
	}
	registry := NewFormats()
	type layout struct {
		DateTime *DateTime `json:"datetime,omitempty"`
	}
	for i := range testCases {
		tc := testCases[i]
		t.Run(tc.Name, func(t *testing.T) {
			test := new(layout)
			cfg := &mapstructure.DecoderConfig{
				DecodeHook:       registry.MapStructureHookFunc(),
				WeaklyTypedInput: false,
				Result:           test,
			}
			d, err := mapstructure.NewDecoder(cfg)
			assert.Nil(t, err)
			input := make(map[string]interface{})
			input["datetime"] = tc.Input
			err = d.Decode(input)
			assert.Error(t, err, "error expected got none")
		})
	}
}
