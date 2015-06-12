/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package uid

import (
	"fmt"
	"strconv"
	"strings"

	"code.google.com/p/go-uuid/uuid"
	log "github.com/golang/glog"
)

type UID struct {
	group uint64
	name  string
	ser   string
}

func New(group uint64, name string) *UID {
	if name == "" {
		name = uuid.New()
	}
	return &UID{
		group: group,
		name:  name,
		ser:   fmt.Sprintf("%x_%s", group, name),
	}
}

func (self *UID) Name() string {
	if self != nil {
		return self.name
	}
	return ""
}

func (self *UID) Group() uint64 {
	if self != nil {
		return self.group
	}
	return 0
}

func (self *UID) String() string {
	if self != nil {
		return self.ser
	}
	return ""
}

func Parse(ser string) *UID {
	parts := strings.SplitN(ser, "_", 2)
	if len(parts) != 2 {
		return nil
	}
	group, err := strconv.ParseUint(parts[0], 16, 64)
	if err != nil {
		log.Errorf("illegal UID group %q: %v", parts[0], err)
		return nil
	}
	if parts[1] == "" {
		log.Errorf("missing UID name: %q", ser)
		return nil
	}
	return &UID{
		group: group,
		name:  parts[1],
		ser:   ser,
	}
}
