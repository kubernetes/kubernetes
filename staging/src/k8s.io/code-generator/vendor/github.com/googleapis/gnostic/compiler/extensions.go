// Copyright 2017 Google LLC. All Rights Reserved.
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

package compiler

import (
	"bytes"
	"fmt"
	"os/exec"
	"strings"

	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes/any"
	extensions "github.com/googleapis/gnostic/extensions"
	yaml "gopkg.in/yaml.v3"
)

// ExtensionHandler describes a binary that is called by the compiler to handle specification extensions.
type ExtensionHandler struct {
	Name string
}

// CallExtension calls a binary extension handler.
func CallExtension(context *Context, in *yaml.Node, extensionName string) (handled bool, response *any.Any, err error) {
	if context == nil || context.ExtensionHandlers == nil {
		return false, nil, nil
	}
	handled = false
	for _, handler := range *(context.ExtensionHandlers) {
		response, err = handler.handle(in, extensionName)
		if response == nil {
			continue
		} else {
			handled = true
			break
		}
	}
	return handled, response, err
}

func (extensionHandlers *ExtensionHandler) handle(in *yaml.Node, extensionName string) (*any.Any, error) {
	if extensionHandlers.Name != "" {
		yamlData, _ := yaml.Marshal(in)
		request := &extensions.ExtensionHandlerRequest{
			CompilerVersion: &extensions.Version{
				Major: 0,
				Minor: 1,
				Patch: 0,
			},
			Wrapper: &extensions.Wrapper{
				Version:       "unknown", // TODO: set this to the type/version of spec being parsed.
				Yaml:          string(yamlData),
				ExtensionName: extensionName,
			},
		}
		requestBytes, _ := proto.Marshal(request)
		cmd := exec.Command(extensionHandlers.Name)
		cmd.Stdin = bytes.NewReader(requestBytes)
		output, err := cmd.Output()
		if err != nil {
			return nil, err
		}
		response := &extensions.ExtensionHandlerResponse{}
		err = proto.Unmarshal(output, response)
		if err != nil || !response.Handled {
			return nil, err
		}
		if len(response.Errors) != 0 {
			return nil, fmt.Errorf("Errors when parsing: %+v for field %s by vendor extension handler %s. Details %+v", in, extensionName, extensionHandlers.Name, strings.Join(response.Errors, ","))
		}
		return response.Value, nil
	}
	return nil, nil
}
