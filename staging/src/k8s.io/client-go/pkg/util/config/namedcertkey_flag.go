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
	"flag"
	"strings"
)

// NamedCertKey is a flag value parsing "certfile,keyfile" and "certfile,keyfile:name,name,name".
type NamedCertKey struct {
	Names             []string
	CertFile, KeyFile string
}

var _ flag.Value = &NamedCertKey{}

func (nkc *NamedCertKey) String() string {
	s := nkc.CertFile + "," + nkc.KeyFile
	if len(nkc.Names) > 0 {
		s = s + ":" + strings.Join(nkc.Names, ",")
	}
	return s
}

func (nkc *NamedCertKey) Set(value string) error {
	cs := strings.SplitN(value, ":", 2)
	var keycert string
	if len(cs) == 2 {
		var names string
		keycert, names = strings.TrimSpace(cs[0]), strings.TrimSpace(cs[1])
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
		return errors.New("expected comma separated certificate and key file paths")
	}
	nkc.CertFile = strings.TrimSpace(cs[0])
	nkc.KeyFile = strings.TrimSpace(cs[1])
	return nil
}

func (*NamedCertKey) Type() string {
	return "namedCertKey"
}

// NamedCertKeyArray is a flag value parsing NamedCertKeys, each passed with its own
// flag instance (in contrast to comma separated slices).
type NamedCertKeyArray struct {
	value   *[]NamedCertKey
	changed bool
}

var _ flag.Value = &NamedCertKey{}

// NewNamedKeyCertArray creates a new NamedCertKeyArray with the internal value
// pointing to p.
func NewNamedCertKeyArray(p *[]NamedCertKey) *NamedCertKeyArray {
	return &NamedCertKeyArray{
		value: p,
	}
}

func (a *NamedCertKeyArray) Set(val string) error {
	nkc := NamedCertKey{}
	err := nkc.Set(val)
	if err != nil {
		return err
	}
	if !a.changed {
		*a.value = []NamedCertKey{nkc}
		a.changed = true
	} else {
		*a.value = append(*a.value, nkc)
	}
	return nil
}

func (a *NamedCertKeyArray) Type() string {
	return "namedCertKey"
}

func (a *NamedCertKeyArray) String() string {
	nkcs := make([]string, 0, len(*a.value))
	for i := range *a.value {
		nkcs = append(nkcs, (*a.value)[i].String())
	}
	return "[" + strings.Join(nkcs, ";") + "]"
}
