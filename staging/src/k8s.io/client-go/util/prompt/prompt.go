/*
Copyright 2017 The Kubernetes Authors.

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

package prompt

import (
	"fmt"
	"io"

	"github.com/howeyc/gopass"
)

const (
	ShowEcho     = true
	DontShowEcho = false
	Mask         = true
	DontMask     = false
)

type Prompter struct {
	reader io.Reader
}

func NewPrompter(reader io.Reader) *Prompter {
	return &Prompter{reader: reader}
}

func (p Prompter) Prompt(field string, showEcho bool, mask bool) (result string, err error) {
	fmt.Printf("Please enter %s: ", field)
	if showEcho {
		_, err = fmt.Fscan(p.reader, &result)
	} else {
		var data []byte
		if mask {
			data, err = gopass.GetPasswdMasked()
		} else {
			data, err = gopass.GetPasswd()
		}
		result = string(data)
	}
	return result, err
}
