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

package yamlpc

import (
	"io"
	"io/ioutil"

	"github.com/go-openapi/runtime"

	"gopkg.in/yaml.v2"
)

// YAMLConsumer creates a consumer for yaml data
func YAMLConsumer() runtime.Consumer {
	return runtime.ConsumerFunc(func(r io.Reader, v interface{}) error {
		buf, err := ioutil.ReadAll(r)
		if err != nil {
			return err
		}
		return yaml.Unmarshal(buf, v)
	})
}

// YAMLProducer creates a producer for yaml data
func YAMLProducer() runtime.Producer {
	return runtime.ProducerFunc(func(w io.Writer, v interface{}) error {
		b, _ := yaml.Marshal(v) // can't make this error come up
		_, err := w.Write(b)
		return err
	})
}
