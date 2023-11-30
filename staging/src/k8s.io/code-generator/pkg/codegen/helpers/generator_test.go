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

package helpers_test

import (
	"crypto/sha256"
	"encoding/hex"
	goflag "flag"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"k8s.io/code-generator/pkg/codegen/helpers"
	"k8s.io/code-generator/pkg/fs"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	klogtest "k8s.io/klog/v2/test"
)

func TestGenerate(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test in short mode")
	}
	currPkg := reflect.TypeOf(empty{}).PkgPath()
	if !strings.HasPrefix(currPkg, "k8s.io/code-generator") {
		t.Skipf("skipping test in vendored package %s", currPkg)
	}

	_ = klogtest.InitKlog(t)
	klog.SetLogger(ktesting.NewLogger(t, ktesting.NewConfig()))

	workdir, wantDigest := prepareWorkdir(t)

	args := &helpers.Args{
		InputDir: workdir,
	}
	if err := args.Validate(); err != nil {
		t.Fatal(err)
	}

	// Run the generator.
	gen := helpers.Generator{
		Flags: goflag.NewFlagSet("test", goflag.ContinueOnError),
	}
	if err := gen.Generate(args); err != nil {
		t.Fatal(err)
	}

	// Check the output.
	if gotDigest, err := hashdir(workdir); err != nil {
		t.Fatal(err)
	} else if gotDigest != wantDigest {
		t.Errorf("got %q, want %q", gotDigest, wantDigest)
	}
}

func prepareWorkdir(tb testing.TB) (string, string) {
	var sourcedir string
	if root, err := fs.RootDir(); err != nil {
		tb.Fatal(err)
	} else {
		sourcedir = path.Join(root, "examples")
	}
	if rd, err := filepath.EvalSymlinks(sourcedir); err != nil {
		tb.Fatal(err)
	} else {
		sourcedir = rd
	}
	if _, err := os.Stat(sourcedir); err != nil {
		tb.Fatal(err)
	}
	if err := restoreSources(sourcedir); err != nil {
		tb.Fatal(err)
	}
	var initDigest string
	if d, err := hashdir(sourcedir); err != nil {
		tb.Fatal(err)
	} else {
		initDigest = d
	}

	return sourcedir, initDigest
}

func restoreSources(sourcedir string) error {
	klog.V(5).Infof("Restoring %s", sourcedir)
	err := fs.WithinDirectory(sourcedir, func() error {
		klog.V(5).Infof("Calling `git checkout .`")
		if err := exec.Command("git", "checkout", ".").Run(); err != nil {
			return err
		}
		klog.V(5).Infof("Calling `git clean -fdx .`")
		if err := exec.Command("git", "clean", "-fdx", ".").Run(); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		klog.Errorf("Error restoring %s: %#v", sourcedir, err)
	}
	return err
}

// hashdir returns a sha256 hex encoded digest of the directory.
func hashdir(dir string) (string, error) {
	hash := sha256.New()
	if err := filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if info.IsDir() {
			return nil
		}
		data, err1 := os.ReadFile(p)
		if err1 != nil {
			return err1
		}
		if _, err1 := hash.Write(data); err1 != nil {
			return err1
		}
		return nil
	}); err != nil {
		return "", err
	}
	// as hex encoded string
	return hex.EncodeToString(hash.Sum(nil)), nil
}

type empty struct{}
