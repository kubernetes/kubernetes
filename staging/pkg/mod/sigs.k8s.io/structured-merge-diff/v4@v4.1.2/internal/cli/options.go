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
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

var (
	ErrTooManyOperations = errors.New("exactly one of --merge, --compare, --validate or --fieldset must be provided")
	ErrNeedTwoArgs       = errors.New("--merge and --compare require both --lhs and --rhs")
)

type Options struct {
	schemaPath string
	typeName   string

	output string

	// options determining the operation to perform
	listTypes    bool
	validatePath string
	merge        bool
	compare      bool
	fieldset     string

	// arguments for merge or compare
	lhsPath string
	rhsPath string
}

func (o *Options) AddFlags(fs *flag.FlagSet) {
	fs.StringVar(&o.schemaPath, "schema", "", "Path to the schema file for this operation. Required.")
	fs.StringVar(&o.typeName, "type-name", "", "Name of type in the schema to use. If empty, the first type in the schema will be used.")

	fs.StringVar(&o.output, "output", "-", "Output location (if the command has output). '-' means stdout.")

	// The three supported operations. We could make these into subcommands
	// and that would probably make more sense, but this is easy and this
	// binary is mostly just to enable a little exploration, so this is
	// fine for now.
	fs.BoolVar(&o.listTypes, "list-types", false, "List all the types in the schema and exit.")
	fs.StringVar(&o.validatePath, "validate", "", "Path to a file to perform a validation operation on.")
	fs.BoolVar(&o.merge, "merge", false, "Perform a merge operation between --lhs and --rhs")
	fs.BoolVar(&o.compare, "compare", false, "Perform a compare operation between --lhs and --rhs")
	fs.StringVar(&o.fieldset, "fieldset", "", "Path to a file for which we should build a fieldset.")

	fs.StringVar(&o.lhsPath, "lhs", "", "Path to a file containing the left hand side of the operation")
	fs.StringVar(&o.rhsPath, "rhs", "", "Path to a file containing the right hand side of the operation")
}

// resolve turns options in to an operation that can be executed.
func (o *Options) Resolve() (Operation, error) {
	var base operationBase
	if o.schemaPath == "" {
		return nil, errors.New("a schema is required")
	}
	b, err := ioutil.ReadFile(o.schemaPath)
	if err != nil {
		return nil, fmt.Errorf("unable to read schema %q: %v", o.schemaPath, err)
	}
	base.parser, err = typed.NewParser(typed.YAMLObject(b))
	if err != nil {
		return nil, fmt.Errorf("schema %q has errors:\n%v", o.schemaPath, err)
	}

	if o.typeName == "" {
		types := base.parser.Schema.Types
		if len(types) == 0 {
			return nil, errors.New("no types were given in the schema")
		}
		base.typeName = types[0].Name
	} else {
		base.typeName = o.typeName
	}

	// Count how many operations were requested
	c := map[bool]int{true: 1}
	count := c[o.merge] + c[o.compare] + c[o.validatePath != ""] + c[o.listTypes] + c[o.fieldset != ""]
	if count > 1 {
		return nil, ErrTooManyOperations
	}

	switch {
	case o.listTypes:
		return listTypes{base}, nil
	case o.validatePath != "":
		return validation{base, o.validatePath}, nil
	case o.merge:
		if o.lhsPath == "" || o.rhsPath == "" {
			return nil, ErrNeedTwoArgs
		}
		return merge{base, o.lhsPath, o.rhsPath}, nil
	case o.compare:
		if o.lhsPath == "" || o.rhsPath == "" {
			return nil, ErrNeedTwoArgs
		}
		return compare{base, o.lhsPath, o.rhsPath}, nil
	case o.fieldset != "":
		return fieldset{base, o.fieldset}, nil
	}
	return nil, errors.New("no operation requested")
}

func (o *Options) OpenOutput() (io.WriteCloser, error) {
	if o.output == "-" {
		return os.Stdout, nil
	}
	f, err := os.Create(o.output)
	if err != nil {
		return nil, fmt.Errorf("unable to open %q for writing: %v", o.output, err)
	}
	return f, nil
}
