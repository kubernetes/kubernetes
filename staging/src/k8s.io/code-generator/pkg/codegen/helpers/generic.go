/*
Copyright 2023 The Kubernetes Authors.

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

package helpers

import (
	goflag "flag"
	"fmt"
	"os"
	"path"
	"regexp"

	"github.com/spf13/pflag"
	"k8s.io/code-generator/pkg/fs"
	"k8s.io/code-generator/pkg/fs/gosrc"
	"k8s.io/code-generator/pkg/osbin/golang"
	"k8s.io/klog/v2"
)

type genFn func(fs *pflag.FlagSet, args []string) error

type genConf struct {
	name             string
	fileSuffix       string
	searchPattern    string
	useExtraPeerDirs bool
	genFn
}

func (gc genConf) suffix() string {
	if gc.fileSuffix == "" {
		return gc.name
	}
	return gc.fileSuffix
}

func zzGeneratedPathspec(name string) string {
	return fmt.Sprintf("zz_generated.%s.go", name)
}

func (g *Generator) generateGen(args *Args, gc genConf) error {
	return fs.WithinDirectory(args.InputDir, func() error {
		pkgs, err := collectPackages(args, gc)
		if err != nil {
			return err
		}
		klog.V(2).Infof("Found %d packages with %s-gen tags",
			len(pkgs), gc.name)
		if len(pkgs) > 0 {
			if err = deleteGenerated(args, gc); err != nil {
				return err
			}
		}
		if err := g.generatePackages(args, gc, pkgs); err != nil {
			return err
		}

		klog.V(2).Infof("%s-gen completed successfully", gc.name)

		return nil
	})
}

func collectPackages(args *Args, conf genConf) ([]string, error) {
	var inputPkgs = make(map[string]bool, 1)
	matcher := gosrc.FileContains(regexp.MustCompile(
		regexp.QuoteMeta(conf.searchPattern),
	))
	if files, err := gosrc.Find(args.InputDir, matcher); err != nil {
		return nil, err
	} else {
		klog.V(3).Infof("Found %d files with %s-gen tags",
			len(files), conf.name)
		klog.V(4).Infof("Files with %s-gen tags: %#v",
			conf.name, files)

		for _, file := range files {
			klog.V(4).Infof("Resolving package for %s", file)
			if p, errr := resolvePackage(file); errr != nil {
				klog.Errorf("Error finding package for %s: %s", file, errr)
				return nil, errr
			} else {
				klog.V(4).Infof("Found package %s", p)
				inputPkgs[p] = true
			}
		}
	}
	var pkgs []string
	for p := range inputPkgs {
		pkgs = append(pkgs, p)
	}
	return pkgs, nil
}

func deleteGenerated(args *Args, gc genConf) error {
	pathspec := zzGeneratedPathspec(gc.suffix())
	if genFiles, err := gosrc.Find(args.InputDir, gosrc.FileEndsWith(pathspec)); err != nil {
		return err
	} else {
		klog.V(2).Infof("Deleting %d existing %s helpers",
			len(genFiles), gc.name)
		for i := len(genFiles) - 1; i >= 0; i-- {
			file := genFiles[i]
			if _, oserr := os.Stat(file); oserr != nil && os.IsNotExist(oserr) {
				continue
			}
			klog.V(4).Infof("Deleting %s", file)
			if err = os.Remove(file); err != nil {
				return err
			}
		}
	}
	return nil
}

func (g *Generator) generatePackages(args *Args, gc genConf, pkgs []string) error {
	if len(pkgs) == 0 {
		return nil
	}
	klog.Infof("Generating %s code for %d targets",
		gc.name, len(pkgs))

	dcargs := []string{
		"--output-file", zzGeneratedPathspec(gc.suffix()),
		"--go-header-file", args.Boilerplate,
	}
	dcargs = append(dcargs, pkgs...)
	if gc.useExtraPeerDirs {
		for _, pkg := range args.ExtraPeerDirs {
			dcargs = append(dcargs, "--extra-peer-dirs", pkg)
		}
	}
	genName := gc.name + "-gen"
	fl := pflag.NewFlagSet(genName, pflag.ContinueOnError)
	fl.AddGoFlagSet(g.flags()) // make sure we get the klog flags
	klog.V(3).Infof("Running %s with args %q",
		genName, dcargs)

	return asBinary(genName, func() error {
		return gc.genFn(fl, dcargs)
	})
}

func (g *Generator) flags() *goflag.FlagSet {
	if g.Flags == nil {
		return goflag.CommandLine
	}
	return g.Flags
}

func resolvePackage(file string) (string, error) {
	dir := path.Dir(file)
	var foundPkg string
	if err := fs.WithinDirectory(dir, func() error {
		if p, err := golang.PackageOf("."); err != nil {
			klog.Errorf("Error finding package for %s: %s", dir, err)
			return err
		} else {
			foundPkg = p
			return nil
		}
	}); err != nil {
		return "", err
	}
	return foundPkg, nil
}

// asBinary runs the given function as if it were the given binary.
// It sets os.Args[0] to binName, and restores it when done. This is required
// so the generator can identify itself as it was executed directly.
func asBinary(binName string, fn func() error) error {
	curr := os.Args[0]
	defer func() {
		os.Args[0] = curr
	}()
	os.Args[0] = binName
	return fn()
}
