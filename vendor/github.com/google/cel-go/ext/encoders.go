// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ext

import (
	"encoding/base64"
	"reflect"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// Encoders returns a cel.EnvOption to configure extended functions for string, byte, and object
// encodings.
//
// Base64.Decode
//
// Decodes base64-encoded string to bytes.
//
// This function will return an error if the string input is not base64-encoded.
//
//     base64.decode(<string>) -> <bytes>
//
// Examples:
//
//     base64.decode('aGVsbG8=')  // return b'hello'
//     base64.decode('aGVsbG8')   // error
//
// Base64.Encode
//
// Encodes bytes to a base64-encoded string.
//
//     base64.encode(<bytes>)  -> <string>
//
// Examples:
//
//     base64.encode(b'hello') // return b'aGVsbG8='
func Encoders() cel.EnvOption {
	return cel.Lib(encoderLib{})
}

type encoderLib struct{}

func (encoderLib) CompileOptions() []cel.EnvOption {
	return []cel.EnvOption{
		cel.Function("base64.decode",
			cel.Overload("base64_decode_string", []*cel.Type{cel.StringType}, cel.BytesType,
				cel.UnaryBinding(func(str ref.Val) ref.Val {
					s := str.(types.String)
					return bytesOrError(base64DecodeString(string(s)))
				}))),
		cel.Function("base64.encode",
			cel.Overload("base64_encode_bytes", []*cel.Type{cel.BytesType}, cel.StringType,
				cel.UnaryBinding(func(bytes ref.Val) ref.Val {
					b := bytes.(types.Bytes)
					return stringOrError(base64EncodeBytes([]byte(b)))
				}))),
	}
}

func (encoderLib) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

func base64DecodeString(str string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(str)
}

func base64EncodeBytes(bytes []byte) (string, error) {
	return base64.StdEncoding.EncodeToString(bytes), nil
}

var (
	bytesListType = reflect.TypeOf([]byte{})
)
