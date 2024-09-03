/*
Copyright 2015 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/tools/imports"

	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

// ExecuteTargets runs the generators for the provided targets.
func (c *Context) ExecuteTargets(targets []Target) error {
	klog.V(5).Infof("ExecuteTargets: %d targets", len(targets))

	var errs []error
	for _, tgt := range targets {
		if err := c.ExecuteTarget(tgt); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("some targets had errors: %w", errors.Join(errs...))
	}
	return nil
}

type DefaultFileType struct {
	Format   func([]byte) ([]byte, error)
	Assemble func(io.Writer, *File)
}

func (ft DefaultFileType) AssembleFile(f *File, pathname string) error {
	klog.V(5).Infof("Assembling file %q", pathname)

	destFile, err := os.Create(pathname)
	if err != nil {
		return err
	}
	defer destFile.Close()

	b := &bytes.Buffer{}
	et := NewErrorTracker(b)
	ft.Assemble(et, f)
	if et.Error() != nil {
		return et.Error()
	}
	if formatted, err := ft.Format(b.Bytes()); err != nil {
		err = fmt.Errorf("unable to format file %q (%v)", pathname, err)
		// Write the file anyway, so they can see what's going wrong and fix the generator.
		if _, err2 := destFile.Write(b.Bytes()); err2 != nil {
			return err2
		}
		return err
	} else {
		_, err = destFile.Write(formatted)
		return err
	}
}

func assembleGoFile(w io.Writer, f *File) {
	w.Write(f.Header)
	fmt.Fprintf(w, "package %v\n\n", f.PackageName)

	if len(f.Imports) > 0 {
		fmt.Fprint(w, "import (\n")
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

func importsWrapper(src []byte) ([]byte, error) {
	opt := imports.Options{
		Comments:   true,
		TabIndent:  true,
		TabWidth:   8,
		FormatOnly: true, // Disable the insertion and deletion of imports
	}
	return imports.Process("", src, &opt)
}

func NewGoFile() *DefaultFileType {
	return &DefaultFileType{
		Format:   importsWrapper,
		Assemble: assembleGoFile,
	}
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

// ExecuteTarget runs the generators for a single target.
func (c *Context) ExecuteTarget(tgt Target) error {
	tgtDir := tgt.Dir()
	if tgtDir == "" {
		return fmt.Errorf("no directory for target %s", tgt.Path())
	}
	klog.V(5).Infof("Executing target %q (%q)", tgt.Name(), tgtDir)

	// Filter out any types the *package* doesn't care about.
	packageContext := c.filteredBy(tgt.Filter)

	if err := os.MkdirAll(tgtDir, 0755); err != nil {
		return err
	}

	files := map[string]*File{}
	for _, g := range tgt.Generators(packageContext) {
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
				PackageName: tgt.Name(),
				PackagePath: tgt.Path(),
				PackageDir:  tgt.Dir(),
				Header:      tgt.Header(g.Filename()),
				Imports:     map[string]struct{}{},
			}
			files[f.Name] = f
		} else if f.FileType != g.FileType() {
			return fmt.Errorf("file %q already has type %q, but generator %q wants to use type %q", f.Name, f.FileType, g.Name(), g.FileType())
		}

		if vars := g.PackageVars(genContext); len(vars) > 0 {
			addIndentHeaderComment(&f.Vars, "Package-wide variables from generator %q.", g.Name())
			for _, v := range vars {
				if _, err := fmt.Fprintf(&f.Vars, "%s\n", v); err != nil {
					return err
				}
			}
		}
		if consts := g.PackageConsts(genContext); len(consts) > 0 {
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

	var errs []error
	for _, f := range files {
		finalPath := filepath.Join(tgtDir, f.Name)
		assembler, ok := c.FileTypes[f.FileType]
		if !ok {
			return fmt.Errorf("the file type %q registered for file %q does not exist in the context", f.FileType, f.Name)
		}
		if err := assembler.AssembleFile(f, finalPath); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("errors in target %q: %w", tgt.Path(), errors.Join(errs...))
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
	if err := generator.Finalize(c, et); err != nil {
		return err
	}
	return et.Error()
}
