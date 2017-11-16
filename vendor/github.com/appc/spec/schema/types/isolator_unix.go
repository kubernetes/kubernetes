// Copyright 2016 The appc Authors
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

package types

import (
	"encoding/json"
)

var (
	UnixIsolatorNames = make(map[ACIdentifier]struct{})
)

const (
	//TODO(lucab): add "ulimit" isolators
	UnixSysctlName = "os/unix/sysctl"
)

func init() {
	for name, con := range map[ACIdentifier]IsolatorValueConstructor{
		UnixSysctlName: func() IsolatorValue { return &UnixSysctl{} },
	} {
		AddIsolatorName(name, UnixIsolatorNames)
		AddIsolatorValueConstructor(name, con)
	}
}

type UnixSysctl map[string]string

func (s *UnixSysctl) UnmarshalJSON(b []byte) error {
	var v map[string]string
	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}
	*s = UnixSysctl(v)
	return err
}

func (s UnixSysctl) AssertValid() error {
	return nil
}

func (s UnixSysctl) multipleAllowed() bool {
	return false
}
func (s UnixSysctl) Conflicts() []ACIdentifier {
	return nil
}

func (s UnixSysctl) AsIsolator() Isolator {
	isol := isolatorMap[UnixSysctlName]()

	b, err := json.Marshal(s)
	if err != nil {
		panic(err)
	}
	valRaw := json.RawMessage(b)
	return Isolator{
		Name:     UnixSysctlName,
		ValueRaw: &valRaw,
		value:    isol,
	}
}

func NewUnixSysctlIsolator(cfg map[string]string) (*UnixSysctl, error) {
	s := UnixSysctl(cfg)
	if err := s.AssertValid(); err != nil {
		return nil, err
	}
	return &s, nil
}
