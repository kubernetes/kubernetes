/*
Copyright 2015 Google Inc. All rights reserved.

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

package util

import (
	"fmt"
	"strconv"
	"strings"
)

type PortRange struct {
	Base int
	Size int
}

func (pr *PortRange) Contains(p int) bool {
	return (p >= pr.Base) && ((p - pr.Base) < pr.Size)
}

func (pr PortRange) String() string {
	if pr.Size == 0 {
		return ""
	}
	return fmt.Sprintf("[%d-%d]", pr.Base, pr.Base+pr.Size-1)
}

func (pr *PortRange) Set(value string) error {
	value = strings.TrimSpace(value)

	// TODO: Accept "80" syntax
	// TODO: Accept "80+8" syntax

	if value == "" {
		pr.Base = 0
		pr.Size = 0
		return nil
	}

	hyphenIndex := strings.Index(value, "-")
	if hyphenIndex == -1 {
		return fmt.Errorf("Expected hyphen (-) in port range")
	}

	var err error
	var low int
	var high int
	low, err = strconv.Atoi(value[:hyphenIndex])
	if err == nil {
		high, err = strconv.Atoi(value[hyphenIndex+1:])
	}
	if err != nil {
		return fmt.Errorf("Unable to parse port range: %s", value)
	}

	if high < low {
		return fmt.Errorf("Port range end port cannot be less than start port: %s", value)
	}
	pr.Base = low
	pr.Size = 1 + high - low
	return nil
}

func (*PortRange) Type() string {
	return "portRange"
}

func ParsePortRange(value string) (*PortRange, error) {
	pr := &PortRange{}
	err := pr.Set(value)
	if err != nil {
		return nil, err
	}
	return pr, nil
}
