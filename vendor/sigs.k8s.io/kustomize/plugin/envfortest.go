// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package plugin

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/pgmconfig"
	"sigs.k8s.io/kustomize/pkg/plugins"
)

// EnvForTest manages the plugin test environment.
// It sets/resets XDG_CONFIG_HOME, makes/removes a temp objRoot.
type EnvForTest struct {
	t        *testing.T
	compiler *plugins.Compiler
	workDir  string
	oldXdg   string
	wasSet   bool
}

func NewEnvForTest(t *testing.T) *EnvForTest {
	return &EnvForTest{t: t}
}

func (x *EnvForTest) Set() *EnvForTest {
	x.createWorkDir()
	x.compiler = x.makeCompiler()
	x.setEnv()
	return x
}

func (x *EnvForTest) Reset() {
	x.resetEnv()
	x.removeWorkDir()
}

func (x *EnvForTest) BuildGoPlugin(g, v, k string) {
	err := x.compiler.Compile(g, v, k)
	if err != nil {
		x.t.Errorf("compile failed: %v", err)
	}
}

func (x *EnvForTest) BuildExecPlugin(g, v, k string) {
	lowK := strings.ToLower(k)
	obj := filepath.Join(x.compiler.ObjRoot(), g, v, lowK, k)
	src := filepath.Join(x.compiler.SrcRoot(), g, v, lowK, k)
	if err := os.MkdirAll(filepath.Dir(obj), 0755); err != nil {
		x.t.Errorf("error making directory: %s", filepath.Dir(obj))
	}
	cmd := exec.Command("cp", src, obj)
	cmd.Env = os.Environ()
	if err := cmd.Run(); err != nil {
		x.t.Errorf("error copying %s to %s: %v", src, obj, err)
	}
}

func (x *EnvForTest) makeCompiler() *plugins.Compiler {
	// The plugin loader wants to find object code under
	//    $XDG_CONFIG_HOME/kustomize/plugins
	// and the compiler writes object code to
	//    $objRoot
	// so set things up accordingly.
	objRoot := filepath.Join(
		x.workDir, pgmconfig.ProgramName, pgmconfig.PluginRoot)
	err := os.MkdirAll(objRoot, os.ModePerm)
	if err != nil {
		x.t.Error(err)
	}
	srcRoot, err := plugins.DefaultSrcRoot()
	if err != nil {
		x.t.Error(err)
	}
	return plugins.NewCompiler(srcRoot, objRoot)
}

func (x *EnvForTest) createWorkDir() {
	var err error
	x.workDir, err = ioutil.TempDir("", "kustomize-plugin-tests")
	if err != nil {
		x.t.Errorf("failed to make work dir: %v", err)
	}
}

func (x *EnvForTest) removeWorkDir() {
	err := os.RemoveAll(x.workDir)
	if err != nil {
		x.t.Errorf(
			"removing work dir: %s %v", x.workDir, err)
	}
}

func (x *EnvForTest) setEnv() {
	x.oldXdg, x.wasSet = os.LookupEnv(pgmconfig.XDG_CONFIG_HOME)
	os.Setenv(pgmconfig.XDG_CONFIG_HOME, x.workDir)
}

func (x *EnvForTest) resetEnv() {
	if x.wasSet {
		os.Setenv(pgmconfig.XDG_CONFIG_HOME, x.oldXdg)
	} else {
		os.Unsetenv(pgmconfig.XDG_CONFIG_HOME)
	}
}
