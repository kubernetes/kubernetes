/*
Copyright 2018 The Kubernetes Authors.

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

package cli

import (
	"fmt"
	"io"
	"io/ioutil"

	"sigs.k8s.io/structured-merge-diff/v4/typed"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

type Operation interface {
	Execute(io.Writer) error
}

type operationBase struct {
	parser   *typed.Parser
	typeName string
}

func (b operationBase) parseFile(path string) (tv *typed.TypedValue, err error) {
	bytes, err := ioutil.ReadFile(path)
	if err != nil {
		return tv, fmt.Errorf("unable to read file %q: %v", path, err)
	}
	tv, err = b.parser.Type(b.typeName).FromYAML(typed.YAMLObject(bytes))
	if err != nil {
		return tv, fmt.Errorf("unable to validate file %q:\n%v", path, err)
	}
	return tv, nil
}

type validation struct {
	operationBase

	fileToValidate string
}

func (v validation) Execute(_ io.Writer) error {
	_, err := v.parseFile(v.fileToValidate)
	return err
}

type fieldset struct {
	operationBase

	fileToUse string
}

func (f fieldset) Execute(w io.Writer) error {
	tv, err := f.parseFile(f.fileToUse)
	if err != nil {
		return err
	}

	empty, err := f.parser.Type(f.typeName).FromYAML(typed.YAMLObject("{}"))
	if err != nil {
		return err
	}
	c, err := empty.Compare(tv)
	if err != nil {
		return err
	}

	return c.Added.ToJSONStream(w)
}

type listTypes struct {
	operationBase
}

func (l listTypes) Execute(w io.Writer) error {
	for _, td := range l.parser.Schema.Types {
		fmt.Fprintf(w, "%v\n", td.Name)
	}
	return nil
}

type merge struct {
	operationBase

	lhs string
	rhs string
}

func (m merge) Execute(w io.Writer) error {
	lhs, err := m.parseFile(m.lhs)
	if err != nil {
		return err
	}
	rhs, err := m.parseFile(m.rhs)
	if err != nil {
		return err
	}

	out, err := lhs.Merge(rhs)
	if err != nil {
		return err
	}

	yaml, err := value.ToYAML(out.AsValue())
	if err != nil {
		return err
	}
	_, err = w.Write(yaml)

	return err
}

type compare struct {
	operationBase

	lhs string
	rhs string
}

func (c compare) Execute(w io.Writer) error {
	lhs, err := c.parseFile(c.lhs)
	if err != nil {
		return err
	}
	rhs, err := c.parseFile(c.rhs)
	if err != nil {
		return err
	}

	got, err := lhs.Compare(rhs)
	if err != nil {
		return err
	}

	if got.IsSame() {
		_, err = fmt.Fprint(w, "No difference")
		return err
	}

	// TODO: I think it'd be neat if we actually emitted a machine-readable
	// format.

	_, err = fmt.Fprintf(w, got.String())

	return err
}
