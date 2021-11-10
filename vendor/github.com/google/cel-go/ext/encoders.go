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

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/interpreter/functions"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
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
		cel.Declarations(
			decls.NewFunction("base64.decode",
				decls.NewOverload("base64_decode_string",
					[]*exprpb.Type{decls.String},
					decls.Bytes)),
			decls.NewFunction("base64.encode",
				decls.NewOverload("base64_encode_bytes",
					[]*exprpb.Type{decls.Bytes},
					decls.String)),
		),
	}
}

func (encoderLib) ProgramOptions() []cel.ProgramOption {
	wrappedBase64EncodeBytes := callInBytesOutString(base64EncodeBytes)
	wrappedBase64DecodeString := callInStrOutBytes(base64DecodeString)
	return []cel.ProgramOption{
		cel.Functions(
			&functions.Overload{
				Operator: "base64.decode",
				Unary:    wrappedBase64DecodeString,
			},
			&functions.Overload{
				Operator: "base64_decode_string",
				Unary:    wrappedBase64DecodeString,
			},
			&functions.Overload{
				Operator: "base64.encode",
				Unary:    wrappedBase64EncodeBytes,
			},
			&functions.Overload{
				Operator: "base64_encode_bytes",
				Unary:    wrappedBase64EncodeBytes,
			},
		),
	}
}

func base64DecodeString(str string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(str)
}

func base64EncodeBytes(bytes []byte) (string, error) {
	return base64.StdEncoding.EncodeToString(bytes), nil
}
