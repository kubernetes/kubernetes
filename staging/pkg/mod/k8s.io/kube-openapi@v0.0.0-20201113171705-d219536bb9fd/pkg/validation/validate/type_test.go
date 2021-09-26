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

package validate

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
)

type expectedJSONType struct {
	value                 interface{}
	expectedJSONType      string
	expectedSwaggerFormat string
}

func TestType_schemaInfoForType(t *testing.T) {
	testTypes := []expectedJSONType{
		{
			value:                 []byte("abc"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "byte",
		},
		{
			value:                 strfmt.Date(time.Date(2014, 10, 10, 0, 0, 0, 0, time.UTC)),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "date",
		},
		{
			value:                 strfmt.NewDateTime(),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "date-time",
		},
		{
			value:                 strfmt.URI("http://thisisleadingusnowhere.com"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "uri",
		},
		{
			value:                 strfmt.Email("fred@esasymoney.com"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "email",
		},
		{
			value:                 strfmt.Hostname("www.github.com"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "hostname",
		},
		{
			value:                 strfmt.Password("secret"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "password",
		},
		{
			value:                 strfmt.IPv4("192.168.224.1"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "ipv4",
		},
		{
			value:                 strfmt.IPv6("::1"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "ipv6",
		},
		{
			value:                 strfmt.MAC("01:02:03:04:05:06"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "mac",
		},
		{
			value:                 strfmt.UUID("a8098c1a-f86e-11da-bd1a-00112444be1e"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "uuid",
		},
		{
			value:                 strfmt.UUID3("bcd02e22-68f0-3046-a512-327cca9def8f"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "uuid3",
		},
		{
			value:                 strfmt.UUID4("025b0d74-00a2-4048-bf57-227c5111bb34"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "uuid4",
		},
		{
			value:                 strfmt.UUID5("886313e1-3b8a-5372-9b90-0c9aee199e5d"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "uuid5",
		},
		{
			value:                 strfmt.ISBN("0321751043"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "isbn",
		},
		{
			value:                 strfmt.ISBN10("0321751043"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "isbn10",
		},
		{
			value:                 strfmt.ISBN13("978-0321751041"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "isbn13",
		},
		{
			value:                 strfmt.CreditCard("4111-1111-1111-1111"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "creditcard",
		},
		{
			value:                 strfmt.SSN("111-11-1111"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "ssn",
		},
		{
			value:                 strfmt.HexColor("#FFFFFF"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "hexcolor",
		},
		{
			value:                 strfmt.RGBColor("rgb(255,255,255)"),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "rgbcolor",
		},
		// Numerical values
		{
			value:                 true,
			expectedJSONType:      "boolean",
			expectedSwaggerFormat: "",
		},
		{
			value:                 int8(12),
			expectedJSONType:      "integer",
			expectedSwaggerFormat: "int32",
		},
		{
			value:                 uint8(12),
			expectedJSONType:      "integer",
			expectedSwaggerFormat: "int32",
		},
		{
			value:                 int16(12),
			expectedJSONType:      "integer",
			expectedSwaggerFormat: "int32",
		},
		{
			value:            uint16(12),
			expectedJSONType: "integer",
			// TODO: should be uint32
			expectedSwaggerFormat: "int32",
		},
		{
			value:                 int32(12),
			expectedJSONType:      "integer",
			expectedSwaggerFormat: "int32",
		},
		{
			value:            uint32(12),
			expectedJSONType: "integer",
			// TODO: should be uint32
			expectedSwaggerFormat: "int32",
		},
		{
			value:                 int(12),
			expectedJSONType:      "integer",
			expectedSwaggerFormat: "int64",
		},
		{
			value:            uint(12),
			expectedJSONType: "integer",
			// TODO: should be uint64
			expectedSwaggerFormat: "int64",
		},
		{
			value:                 int64(12),
			expectedJSONType:      "integer",
			expectedSwaggerFormat: "int64",
		},
		{
			value:            uint64(12),
			expectedJSONType: "integer",
			// TODO: should be uint64
			expectedSwaggerFormat: "int64",
		},
		{
			value:            float32(12),
			expectedJSONType: "number",
			// TODO: should be float
			expectedSwaggerFormat: "float32",
		},
		{
			value:            float64(12),
			expectedJSONType: "number",
			// TODO: should be double
			expectedSwaggerFormat: "float64",
		},
		{
			value:                 []string{},
			expectedJSONType:      "array",
			expectedSwaggerFormat: "",
		},
		{
			value:                 expectedJSONType{},
			expectedJSONType:      "object",
			expectedSwaggerFormat: "",
		},
		{
			value:                 map[string]bool{"key": false},
			expectedJSONType:      "object",
			expectedSwaggerFormat: "",
		},
		{
			value:                 "simply a string",
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "",
		},
		{
			// NOTE: Go array returns no JSON type
			value:                 [4]int{1, 2, 4, 4},
			expectedJSONType:      "",
			expectedSwaggerFormat: "",
		},
		{
			value:                 strfmt.Base64("ZWxpemFiZXRocG9zZXk="),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "byte",
		},
		{
			value:                 strfmt.Duration(0),
			expectedJSONType:      stringType,
			expectedSwaggerFormat: "duration",
		},
		/*
			Test case for : case reflect.Interface:
				// What to do here?
				panic("dunno what to do here")
		*/
	}

	v := &typeValidator{}
	for _, x := range testTypes {
		jsonType, swaggerFormat := v.schemaInfoForType(x.value)
		assert.Equal(t, x.expectedJSONType, jsonType)
		assert.Equal(t, x.expectedSwaggerFormat, swaggerFormat)

		jsonType, swaggerFormat = v.schemaInfoForType(&x.value)
		assert.Equal(t, x.expectedJSONType, jsonType)
		assert.Equal(t, x.expectedSwaggerFormat, swaggerFormat)
	}
}
