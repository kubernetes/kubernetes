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

package generator

import (
	"bytes"
	"fmt"
	"go/format"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"
)

// ExecutePackages runs the generatiors for all the passed in packages. 'dir'
// is the base directory in which to make all the packages.
func (c *Context) ExecutePackages(dir string, packages Packages) error {
	for _, p := range packages {
		if err := c.ExecutePackage(dir, p); err != nil {
			return err
		}
	}
	return nil
}

type file struct {
	name        string
	packageName string
	header      []byte
	imports     map[string]struct{}
	vars        bytes.Buffer
	consts      bytes.Buffer
	body        bytes.Buffer
}

func (f *file) assembleToFile(pathname string) error {
	log.Printf("Assembling file %q", pathname)
	destFile, err := os.Create(pathname)
	if err != nil {
		return err
	}
	defer destFile.Close()

	b := &bytes.Buffer{}
	et := NewErrorTracker(b)
	f.assemble(et)
	if et.Error() != nil {
		return et.Error()
	}
	if formatted, err := format.Source(b.Bytes()); err != nil {
		log.Printf("Warning: unable to run gofmt on %q (%v).", pathname, err)
		_, err = destFile.Write(b.Bytes())
		return err
	} else {
		_, err = destFile.Write(formatted)
		return err
	}
}

func (f *file) assemble(w io.Writer) {
	w.Write(f.header)
	fmt.Fprintf(w, "package %v\n\n", f.packageName)

	if len(f.imports) > 0 {
		fmt.Fprint(w, "import (\n")
		for i := range f.imports {
			if strings.Contains(i, "\"") {
				// they included quotes, or are using the
				// `name "path/to/pkg"` format.
				fmt.Fprintf(w, "\t%s\n", i)
			} else {
				fmt.Fprintf(w, "\t%q\n", i)
			}
		}
		fmt.Fprint(w, ")\n\n")
	}

	if f.vars.Len() > 0 {
		fmt.Fprint(w, "var (\n")
		w.Write(f.vars.Bytes())
		fmt.Fprint(w, ")\n\n")
	}

	if f.consts.Len() > 0 {
		fmt.Fprint(w, "const (\n")
		w.Write(f.consts.Bytes())
		fmt.Fprint(w, ")\n\n")
	}

	w.Write(f.body.Bytes())
}

// format should be one line only, and not end with \n.
func addIndentHeaderComment(b *bytes.Buffer, format string, args ...interface{}) {
	if b.Len() > 0 {
		fmt.Fprintf(b, "\n\t// "+format+"\n", args...)
	} else {
		fmt.Fprintf(b, "\t// "+format+"\n", args...)
	}
}

func (c *Context) filterTypes(f func(*Context, *types.Type) bool) (filtered []*types.Type) {
	for _, t := range c.Order {
		if f(c, t) {
			filtered = append(filtered, t)
		}
	}
	return
}

func (c *Context) filteredBy(f func(*Context, *types.Type) bool) *Context {
	c2 := *c
	c2.Order = c.filterTypes(f)
	return &c2
}

func (c *Context) addNameSystems(namers namer.NameSystems) {
	if namers == nil {
		return
	}
	// Copy the existing name systems so we don't corrupt a parent context
	oldNamers := c.Namers
	c.Namers = namer.NameSystems{}
	for k, v := range oldNamers {
		c.Namers[k] = v
	}

	for name, namer := range namers {
		c.Namers[name] = namer
	}
}

// ExecutePackage executes every passed in generator, putting its file in the
// specified directory.  ErrorTracker is used to wrap the io.Writer, so
// generators don't need to check for io errors.
func (c *Context) ExecutePackage(dir string, p Package) error {
	path := filepath.Join(dir, p.Path())
	log.Printf("Executing package %v into %v", p.Name(), path)
	packageContext := c.filteredBy(p.Filter)
	os.MkdirAll(path, 0755)
	files := map[string]*file{}
	for _, g := range p.Generators(packageContext) {
		f := files[g.Filename()]
		// Make a filtered context
		genContext := packageContext.filteredBy(g.Filter)
		// Now add any extra name systems.
		genContext.addNameSystems(g.Namers(genContext))
		if f == nil {
			f = &file{
				name:        g.Filename(),
				packageName: p.Name(),
				header:      p.Header(g.Filename()),
				imports:     map[string]struct{}{},
			}
			files[f.name] = f
		}
		if vars := g.PackageVars(genContext); len(vars) > 0 {
			addIndentHeaderComment(&f.vars, "Package-wide variables from generator %q.", g.Name())
			for _, v := range vars {
				if _, err := fmt.Fprintf(&f.vars, "\t%s\n", v); err != nil {
					return err
				}
			}
		}
		if consts := g.PackageVars(genContext); len(consts) > 0 {
			addIndentHeaderComment(&f.consts, "Package-wide consts from generator %q.", g.Name())
			for _, v := range consts {
				if _, err := fmt.Fprintf(&f.consts, "\t%s\n", v); err != nil {
					return err
				}
			}
		}
		if err := genContext.executeBody(&f.body, g); err != nil {
			return err
		}
		if imports := g.Imports(genContext); len(imports) > 0 {
			for _, i := range imports {
				f.imports[i] = struct{}{}
			}
		}
	}

	for _, f := range files {
		if err := f.assembleToFile(filepath.Join(path, f.name)); err != nil {
			return err
		}
	}
	return nil
}

func (c *Context) executeBody(w io.Writer, generator Generator) error {
	et := NewErrorTracker(w)
	if err := generator.Init(c, et); err != nil {
		return err
	}
	for _, t := range c.Order {
		if err := generator.GenerateType(c, t, et); err != nil {
			return err
		}
	}
	return et.Error()
}
