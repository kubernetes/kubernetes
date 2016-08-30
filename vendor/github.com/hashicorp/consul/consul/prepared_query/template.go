package prepared_query

import (
	"fmt"
	"reflect"
	"regexp"
	"strings"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/hil"
	"github.com/hashicorp/hil/ast"
	"github.com/mitchellh/copystructure"
)

// IsTemplate returns true if the given query is a template.
func IsTemplate(query *structs.PreparedQuery) bool {
	return query.Template.Type != ""
}

// CompiledTemplate is an opaque object that can be used later to render a
// prepared query template.
type CompiledTemplate struct {
	// query keeps a copy of the original query for rendering.
	query *structs.PreparedQuery

	// trees contains a map with paths to string fields in a structure to
	// parsed syntax trees, suitable for later evaluation.
	trees map[string]ast.Node

	// re is the compiled regexp, if they supplied one (this can be nil).
	re *regexp.Regexp
}

// Compile validates a prepared query template and returns an opaque compiled
// object that can be used later to render the template.
func Compile(query *structs.PreparedQuery) (*CompiledTemplate, error) {
	// Make sure it's a type we understand.
	if query.Template.Type != structs.QueryTemplateTypeNamePrefixMatch {
		return nil, fmt.Errorf("Bad Template.Type '%s'", query.Template.Type)
	}

	// Start compile.
	ct := &CompiledTemplate{
		trees: make(map[string]ast.Node),
	}

	// Make a copy of the query to use as the basis for rendering later.
	dup, err := copystructure.Copy(query)
	if err != nil {
		return nil, err
	}
	var ok bool
	ct.query, ok = dup.(*structs.PreparedQuery)
	if !ok {
		return nil, fmt.Errorf("Failed to copy query")
	}

	// Walk over all the string fields in the Service sub-structure and
	// parse them as HIL.
	parse := func(path string, v reflect.Value) error {
		tree, err := hil.Parse(v.String())
		if err != nil {
			return fmt.Errorf("Bad format '%s' in Service%s: %s", v.String(), path, err)
		}

		ct.trees[path] = tree
		return nil
	}
	if err := walk(&ct.query.Service, parse); err != nil {
		return nil, err
	}

	// If they supplied a regexp then compile it.
	if ct.query.Template.Regexp != "" {
		var err error
		ct.re, err = regexp.Compile(ct.query.Template.Regexp)
		if err != nil {
			return nil, fmt.Errorf("Bad Regexp: %s", err)
		}
	}

	// Finally do a test render with the supplied name prefix. This will
	// help catch errors before run time, and this is the most minimal
	// prefix it will be expected to run with. The results might not make
	// sense and create a valid service to lookup, but it should render
	// without any errors.
	if _, err = ct.Render(ct.query.Name); err != nil {
		return nil, err
	}

	return ct, nil
}

// Render takes a compiled template and renders it for the given name. For
// example, if the user looks up foobar.query.consul via DNS then we will call
// this function with "foobar" on the compiled template.
func (ct *CompiledTemplate) Render(name string) (*structs.PreparedQuery, error) {
	// Make it "safe" to render a default structure.
	if ct == nil {
		return nil, fmt.Errorf("Cannot render an uncompiled template")
	}

	// Start with a fresh, detached copy of the original so we don't disturb
	// the prototype.
	dup, err := copystructure.Copy(ct.query)
	if err != nil {
		return nil, err
	}
	query, ok := dup.(*structs.PreparedQuery)
	if !ok {
		return nil, fmt.Errorf("Failed to copy query")
	}

	// Run the regular expression, if provided. We execute on a copy here
	// to avoid internal lock contention because we expect this to be called
	// from multiple goroutines.
	var matches []string
	if ct.re != nil {
		re := ct.re.Copy()
		matches = re.FindStringSubmatch(name)
	}

	// Create a safe match function that can't fail at run time. It will
	// return an empty string for any invalid input.
	match := ast.Function{
		ArgTypes:   []ast.Type{ast.TypeInt},
		ReturnType: ast.TypeString,
		Variadic:   false,
		Callback: func(inputs []interface{}) (interface{}, error) {
			i, ok := inputs[0].(int)
			if ok && i >= 0 && i < len(matches) {
				return matches[i], nil
			} else {
				return "", nil
			}
		},
	}

	// Build up the HIL evaluation context.
	config := &hil.EvalConfig{
		GlobalScope: &ast.BasicScope{
			VarMap: map[string]ast.Variable{
				"name.full": ast.Variable{
					Type:  ast.TypeString,
					Value: name,
				},
				"name.prefix": ast.Variable{
					Type:  ast.TypeString,
					Value: query.Name,
				},
				"name.suffix": ast.Variable{
					Type:  ast.TypeString,
					Value: strings.TrimPrefix(name, query.Name),
				},
			},
			FuncMap: map[string]ast.Function{
				"match": match,
			},
		},
	}

	// Run through the Service sub-structure and evaluate all the strings
	// as HIL.
	eval := func(path string, v reflect.Value) error {
		tree, ok := ct.trees[path]
		if !ok {
			return nil
		}

		hv, ht, err := hil.Eval(tree, config)
		if err != nil {
			return fmt.Errorf("Bad evaluation for '%s' in Service%s: %s", v.String(), path, err)
		}
		if ht != ast.TypeString {
			return fmt.Errorf("Expected Service%s field to be a string, got %s", path, ht)
		}

		v.SetString(hv.(string))
		return nil
	}
	if err := walk(&query.Service, eval); err != nil {
		return nil, err
	}

	return query, nil
}
