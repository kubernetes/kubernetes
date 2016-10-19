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

package config

import (
	"errors"
	"strings"
)

type NamedKeyCert struct {
	Names             []string
	KeyFile, CertFile string
}

func (nkc *NamedKeyCert) String() string {
	s := ""
	if len(nkc.Names) > 0 {
		s = strings.Join(nkc.Names, ",") + ":"
	}
	return s + nkc.KeyFile + "," + nkc.CertFile
}

func (nkc *NamedKeyCert) Set(value string) error {
	cs := strings.SplitN(value, ":", 2)
	var keycert string
	if len(cs) == 2 {
		var names string
		names, keycert = strings.TrimSpace(cs[0]), strings.TrimSpace(cs[1])
		if names == "" {
			return errors.New("empty names list is not allowed")
		}
		nkc.Names = nil
		for _, name := range strings.Split(names, ",") {
			nkc.Names = append(nkc.Names, strings.TrimSpace(name))
		}
	} else {
		nkc.Names = nil
		keycert = strings.TrimSpace(cs[0])
	}
	cs = strings.Split(keycert, ",")
	if len(cs) != 2 {
		return errors.New("expected comma separated key and certificate file paths")
	}
	nkc.KeyFile = strings.TrimSpace(cs[0])
	nkc.CertFile = strings.TrimSpace(cs[1])
	return nil
}

func (*NamedKeyCert) Type() string {
	return "namedKeyCert"
}

type NamedKeyCertArray struct {
	value   *[]NamedKeyCert
	changed bool
}

func NewNamedKeyCertArray(p *[]NamedKeyCert) *NamedKeyCertArray {
	return &NamedKeyCertArray{
		value: p,
	}
}

func (a *NamedKeyCertArray) Set(val string) error {
	nkc := NamedKeyCert{}
	err := nkc.Set(val)
	if err != nil {
		return err
	}
	if !a.changed {
		*a.value = []NamedKeyCert{nkc}
		a.changed = true
	} else {
		*a.value = append(*a.value, nkc)
	}
	return nil
}

func (a *NamedKeyCertArray) Type() string {
	return "namedKeyCert"
}

func (a *NamedKeyCertArray) String() string {
	nkcs := make([]string, 0, len(*a.value))
	for i := range *a.value {
		nkcs = append(nkcs, (*a.value)[i].String())
	}
	return "[" + strings.Join(nkcs, ";") + "]"
}
