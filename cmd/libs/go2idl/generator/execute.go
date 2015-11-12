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

// ExecutePackages runs the generators for every package in 'packages'. 'outDir'
// is the base directory in which to place all the generated packages; it
// should be a physical path on disk, not an import path. e.g.:
// /path/to/home/path/to/gopath/src/
// Each package has its import path already, this will be appended to 'outDir'.
func (c *Context) ExecutePackages(outDir string, packages Packages) error {
	for _, p := range packages {
		if err := c.ExecutePackage(outDir, p); err != nil {
			return err
		}
	}
	return nil
}

type golangFileType struct{}

func (ft golangFileType) AssembleFile(f *File, pathname string) error {
	log.Printf("Assembling file %q", pathname)
	destFile, err := os.Create(pathname)
	if err != nil {
		return err
	}
	defer destFile.Close()

	b := &bytes.Buffer{}
	et := NewErrorTracker(b)
	ft.assemble(et, f)
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

func (ft golangFileType) assemble(w io.Writer, f *File) {
	w.Write(f.Header)
	fmt.Fprintf(w, "package %v\n\n", f.PackageName)

	if len(f.Imports) > 0 {
		fmt.Fprint(w, "import (\n")
		// TODO: sort imports like goimports does.
		for i := range f.Imports {
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

	if f.Vars.Len() > 0 {
		fmt.Fprint(w, "var (\n")
		w.Write(f.Vars.Bytes())
		fmt.Fprint(w, ")\n\n")
	}

	if f.Consts.Len() > 0 {
		fmt.Fprint(w, "const (\n")
		w.Write(f.Consts.Bytes())
		fmt.Fprint(w, ")\n\n")
	}

	w.Write(f.Body.Bytes())
}

// format should be one line only, and not end with \n.
func addIndentHeaderComment(b *bytes.Buffer, format string, args ...interface{}) {
	if b.Len() > 0 {
		fmt.Fprintf(b, "\n// "+format+"\n", args...)
	} else {
		fmt.Fprintf(b, "// "+format+"\n", args...)
	}
}

func (c *Context) filteredBy(f func(*Context, *types.Type) bool) *Context {
	c2 := *c
	c2.Order = []*types.Type{}
	for _, t := range c.Order {
		if f(c, t) {
			c2.Order = append(c2.Order, t)
		}
	}
	return &c2
}

// make a new context; inheret c.Namers, but add on 'namers'. In case of a name
// collision, the namer in 'namers' wins.
func (c *Context) addNameSystems(namers namer.NameSystems) *Context {
	if namers == nil {
		return c
	}
	c2 := *c
	// Copy the existing name systems so we don't corrupt a parent context
	c2.Namers = namer.NameSystems{}
	for k, v := range c.Namers {
		c2.Namers[k] = v
	}

	for name, namer := range namers {
		c2.Namers[name] = namer
	}
	return &c2
}

// ExecutePackage executes a single package. 'outDir' is the base directory in
// which to place the package; it should be a physical path on disk, not an
// import path. e.g.: '/path/to/home/path/to/gopath/src/' The package knows its
// import path already, this will be appended to 'outDir'.
func (c *Context) ExecutePackage(outDir string, p Package) error {
	path := filepath.Join(outDir, p.Path())
	log.Printf("Executing package %v into %v", p.Name(), path)
	// Filter out any types the *package* doesn't care about.
	packageContext := c.filteredBy(p.Filter)
	os.MkdirAll(path, 0755)
	files := map[string]*File{}
	for _, g := range p.Generators(packageContext) {
		// Filter out types the *generator* doesn't care about.
		genContext := packageContext.filteredBy(g.Filter)
		// Now add any extra name systems defined by this generator
		genContext = genContext.addNameSystems(g.Namers(genContext))

		fileType := g.FileType()
		if len(fileType) == 0 {
			return fmt.Errorf("generator %q must specify a file type", g.Name())
		}
		f := files[g.Filename()]
		if f == nil {
			// This is the first generator to reference this file, so start it.
			f = &File{
				Name:        g.Filename(),
				FileType:    fileType,
				PackageName: p.Name(),
				Header:      p.Header(g.Filename()),
				Imports:     map[string]struct{}{},
			}
			files[f.Name] = f
		} else {
			if f.FileType != g.FileType() {
				return fmt.Errorf("file %q already has type %q, but generator %q wants to use type %q", f.Name, f.FileType, g.Name(), g.FileType())
			}
		}

		if vars := g.PackageVars(genContext); len(vars) > 0 {
			addIndentHeaderComment(&f.Vars, "Package-wide variables from generator %q.", g.Name())
			for _, v := range vars {
				if _, err := fmt.Fprintf(&f.Vars, "%s\n", v); err != nil {
					return err
				}
			}
		}
		if consts := g.PackageVars(genContext); len(consts) > 0 {
			addIndentHeaderComment(&f.Consts, "Package-wide consts from generator %q.", g.Name())
			for _, v := range consts {
				if _, err := fmt.Fprintf(&f.Consts, "%s\n", v); err != nil {
					return err
				}
			}
		}
		if err := genContext.executeBody(&f.Body, g); err != nil {
			return err
		}
		if imports := g.Imports(genContext); len(imports) > 0 {
			for _, i := range imports {
				f.Imports[i] = struct{}{}
			}
		}
	}

	for _, f := range files {
		assembler, ok := c.FileTypes[f.FileType]
		if !ok {
			return fmt.Errorf("the file type %q registered for file %q does not exist in the context", f.FileType, f.Name)
		}
		if err := assembler.AssembleFile(f, filepath.Join(path, f.Name)); err != nil {
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
