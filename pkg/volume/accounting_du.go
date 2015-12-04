/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package volume

import (
	"errors"
	"os/exec"
	"strconv"
	"strings"
)

var _ Accountable = &AccountingDu{}

type AccountingDu struct {
	path string
}

func (ad *AccountingDu) GetAccounting() (*Accounting, error) {
	if ad.path == "" {
		return &Accounting{BytesUsed: unknownSize}, errors.New("No volume path defined for disk usage accounting.")
	}
	out, err := exec.Command("du", "-s", ad.path).CombinedOutput()
	if err != nil {
		return &Accounting{BytesUsed: unknownSize}, err
	}
	sbytes := strings.Fields(string(out))[0]
	ibytes, err := strconv.Atoi(sbytes)
	if err != nil {
		return &Accounting{BytesUsed: unknownSize}, err
	}
	return &Accounting{BytesUsed: ibytes}, nil
}

func (ad *AccountingDu) Init(path string) {
	ad.path = path
}
