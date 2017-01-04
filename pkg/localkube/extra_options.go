/*
Copyright 2016 The Kubernetes Authors.

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

package localkube

import (
	"fmt"
	"strings"
)

type ExtraOption struct {
	Component string
	Key       string
	Value     string
}

func (e *ExtraOption) String() string {
	return fmt.Sprintf("%s.%s=%s", e.Component, e.Key, e.Value)
}

type ExtraOptionSlice []ExtraOption

func (es *ExtraOptionSlice) Set(value string) error {
	// The component is the value before the first dot.
	componentSplit := strings.SplitN(value, ".", 2)
	if len(componentSplit) < 2 {
		return fmt.Errorf("Invalid value for ExtraOption flag. Value must contain at least one period: %s", value)
	}

	remainder := strings.Join(componentSplit[1:], "")

	keySplit := strings.SplitN(remainder, "=", 2)
	if len(keySplit) != 2 {
		return fmt.Errorf("Invalid value for ExtraOption flag. Value must contain one equal sign: %s", value)
	}

	e := ExtraOption{
		Component: componentSplit[0],
		Key:       keySplit[0],
		Value:     keySplit[1],
	}
	*es = append(*es, e)
	return nil
}

func (es *ExtraOptionSlice) String() string {
	s := []string{}
	for _, e := range *es {
		s = append(s, e.String())
	}
	return strings.Join(s, " ")
}

func (es *ExtraOptionSlice) Type() string {
	return "ExtraOption"
}
