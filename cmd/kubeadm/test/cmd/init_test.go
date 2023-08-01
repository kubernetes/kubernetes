/*
Copyright 2016 The Kubernetes Authors.

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

package kubeadm

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/lithammer/dedent"

	"k8s.io/apimachinery/pkg/util/version"
)

func runKubeadmInit(t testing.TB, args ...string) (string, string, int, error) {
	t.Helper()
	t.Setenv("KUBEADM_INIT_DRYRUN_DIR", os.TempDir())
	kubeadmPath := getKubeadmPath()
	kubeadmArgs := []string{"init", "--dry-run", "--ignore-preflight-errors=all"}
	kubeadmArgs = append(kubeadmArgs, args...)
	return RunCmd(kubeadmPath, kubeadmArgs...)
}

func getKubeadmVersion() *version.Version {
	kubeadmPath := getKubeadmPath()
	kubeadmArgs := []string{"version", "-o=short"}
	out, _, _, err := RunCmd(kubeadmPath, kubeadmArgs...)
	if err != nil {
		panic(fmt.Sprintf("could not run 'kubeadm version -o=short': %v", err))
	}
	return version.MustParseSemantic(strings.TrimSpace(out))
}

func TestCmdInitToken(t *testing.T) {
	initTest := []struct {
		name     string
		args     string
		expected bool
	}{
		{
			name:     "invalid token size",
			args:     "--token=abcd:1234567890abcd",
			expected: false,
		},
		{
			name:     "invalid token non-lowercase",
			args:     "--token=Abcdef:1234567890abcdef",
			expected: false,
		},
		{
			name:     "valid token is accepted",
			args:     "--token=abcdef.0123456789abcdef",
			expected: true,
		},
	}

	for _, rt := range initTest {
		t.Run(rt.name, func(t *testing.T) {
			_, _, _, err := runKubeadmInit(t, rt.args)
			if (err == nil) != rt.expected {
				t.Fatalf(dedent.Dedent(`
					CmdInitToken test case %q failed with an error: %v
					command 'kubeadm init %s'
						expected: %t
						err: %t
					`),
					rt.name,
					err,
					rt.args,
					rt.expected,
					(err == nil),
				)
			}
		})
	}
}

func TestCmdInitKubernetesVersion(t *testing.T) {
	initTest := []struct {
		name     string
		args     string
		expected bool
	}{
		{
			name:     "invalid semantic version string is detected",
			args:     "--kubernetes-version=v1.1",
			expected: false,
		},
		{
			name:     "valid version is accepted",
			args:     "--kubernetes-version=" + getKubeadmVersion().String(),
			expected: true,
		},
	}

	for _, rt := range initTest {
		t.Run(rt.name, func(t *testing.T) {
			_, _, _, err := runKubeadmInit(t, rt.args)
			if (err == nil) != rt.expected {
				t.Fatalf(dedent.Dedent(`
					CmdInitKubernetesVersion test case %q failed with an error: %v
					command 'kubeadm init %s'
						expected: %t
						err: %t
					`),
					rt.name,
					err,
					rt.args,
					rt.expected,
					(err == nil),
				)
			}
		})
	}
}

func TestCmdInitConfig(t *testing.T) {
	initTest := []struct {
		name     string
		args     string
		expected bool
	}{
		{
			name:     "fail on non existing path",
			args:     "--config=/does/not/exist/foo/bar",
			expected: false,
		},
		{
			name:     "can't load v1beta1 config",
			args:     "--config=testdata/init/v1beta1.yaml",
			expected: false,
		},
		{
			name:     "can't load v1beta2 config",
			args:     "--config=testdata/init/v1beta2.yaml",
			expected: false,
		},
		{
			name:     "can load v1beta3 config",
			args:     "--config=testdata/init/v1beta3.yaml",
			expected: true,
		},
		{
			name:     "don't allow mixed arguments v1beta3",
			args:     "--kubernetes-version=1.11.0 --config=testdata/init/v1beta3.yaml",
			expected: false,
		},
		{
			name:     "can load current component config",
			args:     "--config=testdata/init/current-component-config.yaml",
			expected: true,
		},
		{
			name:     "can't load old component config",
			args:     "--config=testdata/init/old-component-config.yaml",
			expected: false,
		},
	}

	for _, rt := range initTest {
		t.Run(rt.name, func(t *testing.T) {
			_, _, _, err := runKubeadmInit(t, rt.args)
			if (err == nil) != rt.expected {
				t.Fatalf(dedent.Dedent(`
						CmdInitConfig test case %q failed with an error: %v
						command 'kubeadm init %s'
							expected: %t
							err: %t
						`),
					rt.name,
					err,
					rt.args,
					rt.expected,
					(err == nil),
				)
			}
		})
	}
}

func TestCmdInitAPIPort(t *testing.T) {
	initTest := []struct {
		name     string
		args     string
		expected bool
	}{
		{
			name:     "fail on non-string port",
			args:     "--apiserver-bind-port=foobar",
			expected: false,
		},
		{
			name:     "fail on too large port number",
			args:     "--apiserver-bind-port=100000",
			expected: false,
		},
		{
			name:     "fail on negative port number",
			args:     "--apiserver-bind-port=-6000",
			expected: false,
		},
		{
			name:     "accept a valid port number",
			args:     "--apiserver-bind-port=6000",
			expected: true,
		},
	}

	for _, rt := range initTest {
		t.Run(rt.name, func(t *testing.T) {
			_, _, _, err := runKubeadmInit(t, rt.args)
			if (err == nil) != rt.expected {
				t.Fatalf(dedent.Dedent(`
							CmdInitAPIPort test case %q failed with an error: %v
							command 'kubeadm init %s'
								expected: %t
								err: %t
							`),
					rt.name,
					err,
					rt.args,
					rt.expected,
					(err == nil),
				)
			}
		})
	}
}

// TestCmdInitFeatureGates test that feature gates won't make kubeadm panic.
// When go panics it will exit with a 2 code. While we don't expect the init
// calls to succeed in these tests, we ensure that the exit code of calling
// kubeadm with different feature gates is not 2.
func TestCmdInitFeatureGates(t *testing.T) {
	const PanicExitcode = 2

	initTest := []struct {
		name string
		args string
	}{
		{
			name: "no feature gates passed",
			args: "",
		},
		{
			name: "feature gate PublicKeysECDSA=true",
			args: "--feature-gates=PublicKeysECDSA=true",
		},
	}

	for _, rt := range initTest {
		t.Run(rt.name, func(t *testing.T) {
			_, _, exitcode, err := runKubeadmInit(t, rt.args)
			if exitcode == PanicExitcode {
				t.Fatalf(dedent.Dedent(`
							CmdInitFeatureGates test case %q failed with an error: %v
							command 'kubeadm init %s'
                got exit code: %t (panic); unexpected
							`),
					rt.name,
					err,
					rt.args,
					PanicExitcode,
				)
			}
		})
	}
}
