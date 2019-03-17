/*
Copyright 2019 The Kubernetes Authors.

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

package main

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/parser"
	"os"
	"path/filepath"
	"strings"
	"text/template"
	"time"
)

// TODO(mtaufen): There is a lot of cool and advanced stuff in k8s.io/gengo,
// we may want to do something nice with that instead of quick-and-dirty.

// Pass paths to files from the command line
func main() {
	for _, path := range os.Args[1:] {
		if path == "--" {
			continue
		}
		if err := gen(path); err != nil {
			fmt.Fprintln(os.Stderr, "Error:", err)
			os.Exit(1)
		}
	}
}

func gen(path string) error {
	// Read the file and search for +kflag:{type} formatted strings in comments.
	types, err := getTypes(path)
	if err != nil {
		return err
	}

	for _, t := range types {
		// for now we just handle basic, map, or slice

		// TODO(mtaufen): I think this will just ensure it's syntactically correct,
		// but won't necessarily validate whether it's a real type.
		// Can we run the go type checker on these expressions?
		expr, err := parser.ParseExpr(t)
		if err != nil {
			return err
		}

		switch expr := expr.(type) {
		case *ast.ArrayType:
			if err := genSlice(path, t, expr); err != nil {
				return err
			}
		case *ast.MapType:
			if err := genMap(path, t, expr); err != nil {
				return err
			}
		case *ast.Ident: // for some reason they get parsed as Ident instead of BasicLit
			if err := genBasic(path, t); err != nil {
				return err
			}
		default:
			return fmt.Errorf("Unsupported type: %s", t)
		}
	}

	// TODO(mtaufen): Run everything through gofmt.

	return nil
}

func genSlice(path, t string, expr *ast.ArrayType) error {
	// TODO(mtaufen): Implement
	return nil
}

func genMap(path, t string, expr *ast.MapType) error {
	// Extract key/value type names
	ktype := expr.Key.(*ast.Ident).Name
	vtype := expr.Value.(*ast.Ident).Name

	// Uppercase first letter
	kname := strings.ToUpper(ktype[:1]) + ktype[1:]
	vname := strings.ToUpper(vtype[:1]) + vtype[1:]

	data := struct {
		Type string
		Name string
	}{
		Type: t,
		Name: fmt.Sprintf("Map%s%s", kname, vname),
	}

	// create a new file or overwrite existing
	file, err := os.Create(filepath.Join(filepath.Dir(path), fmt.Sprintf("map_%s_%s.go", ktype, vtype)))
	if err != nil {
		return err
	}

	// write templated result into the file
	if err := mapTmpl.Execute(file, data); err != nil {
		return err
	}

	return nil
}

func genBasic(path, t string) error {
	data := struct {
		Type string
		Name string
	}{
		Type: t,
		Name: strings.ToUpper(t[:1]) + t[1:],
	}

	// create a new file or overwrite existing
	file, err := os.Create(filepath.Join(filepath.Dir(path), t+".go"))
	if err != nil {
		return err
	}

	// write templated result into the file
	if err := basicTmpl.Execute(file, data); err != nil {
		return err
	}
	return nil
}

const sentinel = "+kflag:"

func getTypes(path string) ([]string, error) {
	m := map[string]struct{}{}

	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		// entire line must be a comment
		s := strings.TrimSpace(scanner.Text())
		if i := strings.Index(s, "//"); i != 0 {
			continue
		}
		// first thing after the comment starts must be the sentinel (disregarding whitespace)
		s = strings.TrimSpace(s[2:])
		if i := strings.Index(s, sentinel); i != 0 {
			continue
		}
		// there must be nothing else on the line but the type (no more spaces)
		s = strings.TrimSpace(s[len(sentinel):])
		if strings.Contains(s, " ") {
			continue
		}
		// record the requested type
		m[s] = struct{}{}
	}

	// produce a slice of type names
	types := make([]string, len(m))
	i := 0
	for k := range m {
		types[i] = k
		i++
	}

	return types, nil
}

const license = `/*
Copyright {YEAR} The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/`

func licenseWithYear() string {
	return strings.Replace(license, "{YEAR}", time.Now().Format("2006"), 1)
}

const genwarning = "// This file is generated. DO NOT EDIT."

func withHeader(s string) string {
	return fmt.Sprintf("%s\n\n%s\n\n%s", licenseWithYear(), genwarning, s)
}

// TODO(mtaufen): Add documentation comments in templates.

// .Name = Name of the helper. Typically the name of the type with first letter capitalized.
// .Type = Identifier for the type.
var basicTmpl = template.Must(template.New("basic").Parse(withHeader(basicTmplStr)))

const basicTmplStr = `package kflag

import(
	"github.com/spf13/pflag"
)

// {{.Name}}Value contains the scratch space for a registered {{.Type}} flag.
// Values can be applied from this scratch space to a target using the Set or Apply methods.
type {{.Name}}Value struct {
	name string
	value {{.Type}}
	fs *pflag.FlagSet
}

// {{.Name}}Var registers a flag for type {{.Type}} against the FlagSet, and returns a struct
// of type {{.Name}}Value that contains the scratch space the flag will be parsed into.
func (fs *FlagSet) {{.Name}}Var(name string, def {{.Type}}, usage string) *{{.Name}}Value {
	v := &{{.Name}}Value{
		name: name,
		fs: fs.fs,
	}
	fs.fs.{{.Name}}Var(&v.value, name, def, usage)
	return v
}

// Set copies the {{.Type}} value to the target if the flag was detected.
func (v *{{.Name}}Value) Set(target *{{.Type}}) {
	if v.fs.Changed(v.name) {
		*target = v.value
	}
}

// Apply calls the user-provided apply function with the {{.Type}} value if the flag was detected.
func (v *{{.Name}}Value) Apply(apply func(value {{.Type}})) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}
`

const basicTestTmplStr = ``

// TODO(mtaufen): Sadly, maps may not genericize as easily, since anything other
// than map[string]string needs to do string->type conversion... :(
// For example, map_string_bool.go depends on strconv.ParseBool(v)
// May not be that much work to just implement by hand.

// .Name = Name of the helper. Typically like MapKeytypeValuetype.
// .Type = Identifier for the type.
var mapTmpl = template.Must(template.New("basic").Parse(withHeader(mapTmplStr)))

const mapTmplStr = `package kflag

import(
	"fmt"
	"sort"
	"strings"

	"github.com/spf13/pflag"
)

// {{.Name}}Value contains the scratch space for a registered {{.Type}} flag.
// Values can be applied from this scratch space to a target using the Set, Merge, or Apply methods.
type {{.Name}}Value struct {
	name string
	value {{.Type}}
	fs *pflag.FlagSet
}

// {{.Name}}Var registers a flag for type {{.Type}} against the FlagSet, and returns a struct
// of type {{.Name}}Value that contains the scratch space the flag will be parsed into.
func (fs *FlagSet) {{.Name}}Var(name string, def {{.Type}}, sep string, usage string) *{{.Name}}Value {
	val := &{{.Name}}Value{
		name:  name,
		value: make({{.Type}}),
		fs:    fs.fs,
	}
	for k, v := range def {
		val.value[k] = v
	}
	fs.fs.Var(New{{.Name}}(&val.value, sep), name, usage)
	return val
}

// Set copies the map over the target if the flag was detected.
// It completely overwrites any existing target.
func (v *{{.Name}}Value) Set(target *{{.Type}}) {
	if v.fs.Changed(v.name) {
		*target = make({{.Type}})
		for k, v := range v.value {
			(*target)[k] = v
		}
	}
}

// Merge copies the map keys/values piecewise into the target if the flag was detected.
func (v *{{.Name}}Value) Merge(target *{{.Type}}) {
	if v.fs.Changed(v.name) {
		if *target == nil {
			*target = make({{.Type}})
		}
		for k, v := range v.value {
			(*target)[k] = v
		}
	}
}

// Apply calls the user-provided apply function with the map if the flag was detected.
func (v *{{.Name}}Value) Apply(apply func(value {{.Type}})) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}

// TODO(mtaufen): I just copied the below from map_string_string.go, but we can probably
// simplify, e.g. by deduplicating the map values between the below and the scratch space.

// TODO(mtaufen): Consider making all of the below types/methods private, since they should
//  be able to just hide behind this shim now.

// TODO(mtaufen): Consider not exposing sep as an option, and just use = by default and
// manually implement around the couple edge cases (langle_separated_map_string_string.go)

// {{.Name}} can be set from the command line with the format --flag "string=string".
// Multiple flag invocations are supported. For example: --flag "a=foo" --flag "b=bar". If this is desired
// to be the only type invocation NoSplit should be set to true.
// Multiple comma-separated key-value pairs in a single invocation are supported if NoSplit
// is set to false. For example: --flag "a=foo,b=bar".
type {{.Name}} struct {
	Map         *{{.Type}}
	initialized bool
	NoSplit     bool
	sep         string
}

// New{{.Name}} takes a pointer to a {{.Type}} and returns the
// {{.Name}} flag parsing shim for that map.
func New{{.Name}}(m *{{.Type}}, sep string) *{{.Name}} {
	return &{{.Name}}{Map: m, sep: sep}
}

// TODO(mtaufen): figure out how we want to handle "NoSplit".
// Do we want to provide a separate Var constructor for it?
// Do we want to give it a clearer name?
// Do we really want it to be public in the {{.Name}} struct?

// New{{.Name}}NoSplit takes a pointer to a {{.Type}} and sets NoSplit
// value to true and returns the {{.Name}} flag parsing shim for that map.
func New{{.Name}}NoSplit(m *{{.Type}}, sep string) *{{.Name}} {
	return &{{.Name}}{
		Map:     m,
		NoSplit: true,
		sep:     sep,
	}
}

// String implements github.com/spf13/pflag.Value
func (m *{{.Name}}) String() string {
	if m == nil || m.Map == nil {
		return ""
	}
	pairs := []string{}
	for k, v := range *m.Map {
		pairs = append(pairs, fmt.Sprintf("%s%s%s", k, m.sep, v))
	}
	sort.Strings(pairs)
	return strings.Join(pairs, ",")
}

// Set implements github.com/spf13/pflag.Value
func (m *{{.Name}}) Set(value string) error {
	if m.Map == nil {
		return fmt.Errorf("no target (nil pointer to {{.Type}})")
	}
	if !m.initialized || *m.Map == nil {
		// clear default values, or allocate if no existing map
		*m.Map = make({{.Type}})
		m.initialized = true
	}

	// account for comma-separated key-value pairs in a single invocation
	if !m.NoSplit {
		for _, s := range strings.Split(value, ",") {
			if len(s) == 0 {
				continue
			}
			arr := strings.SplitN(s, m.sep, 2)
			if len(arr) != 2 {
				return fmt.Errorf("malformed pair, expect string=string")
			}
			k := strings.TrimSpace(arr[0])
			v := strings.TrimSpace(arr[1])
			(*m.Map)[k] = v
		}
		return nil
	}

	// account for only one key-value pair in a single invocation
	arr := strings.SplitN(value, m.sep, 2)
	if len(arr) != 2 {
		return fmt.Errorf("malformed pair, expect string=string")
	}
	k := strings.TrimSpace(arr[0])
	v := strings.TrimSpace(arr[1])
	(*m.Map)[k] = v
	return nil

}

// Type implements github.com/spf13/pflag.Value
func (*{{.Name}}) Type() string {
	return "{{.Name}}"
}

// Empty implements OmitEmpty
func (m *{{.Name}}) Empty() bool {
	return len(*m.Map) == 0
}
`

const mapTestTmplStr = ``

const sliceTmplStr = ``

const sliceTestTmplStr = ``
