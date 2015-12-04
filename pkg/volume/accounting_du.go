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
	a := NewAccounting()
	if ad.path == "" {
		return a, errors.New("No volume path defined for disk usage accounting.")
	}

	err := ad.runDu(a)
	if err != nil {
		return a, err
	}

	err = ad.runDf(a)
	if err != nil {
		return a, err
	}

	return a, nil
}

func (ad *AccountingDu) runDu(a *Accounting) error {
	out, err := exec.Command("du", "-s", ad.path).CombinedOutput()
	if err != nil {
		return err
	}
	sbytes := strings.Fields(string(out))[0]
	ibytes, err := strconv.Atoi(sbytes)
	if err != nil {
		return err
	}
	a.PodBytesUsed = ibytes
	return nil
}

func (ad *AccountingDu) runDf(a *Accounting) error {
	out, err := exec.Command("df", ad.path).CombinedOutput()
	if err != nil {
		return err
	}
	lines := strings.Split(string(out), "\n")
	header := strings.Fields(lines[0])
	content := strings.Fields(lines[1])

	usedIndex := -1
	availableIndex := -1
	for i, v := range header {
		if v == "Used" {
			usedIndex = i
		}
		if v == "Available" {
			availableIndex = i
		}
	}

	if usedIndex >= 0 {
		used, err := strconv.Atoi(content[usedIndex])
		if err != nil {
			return err
		}
		a.SharedBytesUsed = used
	}

	if availableIndex >= 0 {
		used, err := strconv.Atoi(content[availableIndex])
		if err != nil {
			return err
		}
		a.SharedBytesFree = used
	}
	return nil
}

func (ad *AccountingDu) Init(path string) {
	ad.path = path
}
