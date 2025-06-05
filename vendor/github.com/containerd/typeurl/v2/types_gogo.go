//go:build !no_gogo

/*
   Copyright The containerd Authors.

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

package typeurl

import (
	"reflect"

	gogoproto "github.com/gogo/protobuf/proto"
)

func init() {
	handlers = append(handlers, gogoHandler{})
}

type gogoHandler struct{}

func (gogoHandler) Marshaller(v interface{}) func() ([]byte, error) {
	pm, ok := v.(gogoproto.Message)
	if !ok {
		return nil
	}
	return func() ([]byte, error) {
		return gogoproto.Marshal(pm)
	}
}

func (gogoHandler) Unmarshaller(v interface{}) func([]byte) error {
	pm, ok := v.(gogoproto.Message)
	if !ok {
		return nil
	}

	return func(dt []byte) error {
		return gogoproto.Unmarshal(dt, pm)
	}
}

func (gogoHandler) TypeURL(v interface{}) string {
	pm, ok := v.(gogoproto.Message)
	if !ok {
		return ""
	}
	return gogoproto.MessageName(pm)
}

func (gogoHandler) GetType(url string) (reflect.Type, bool) {
	t := gogoproto.MessageType(url)
	if t == nil {
		return nil, false
	}
	return t.Elem(), true
}
