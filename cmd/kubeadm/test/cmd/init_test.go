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
	"os/exec"
	"strings"
	"testing"

	"github.com/lithammer/dedent"
	"github.com/pkg/errors"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func runKubeadmInit(args ...string) (string, string, int, error) {
	const dryRunDir = "KUBEADM_INIT_DRYRUN_DIR"
	if err := os.Setenv(dryRunDir, os.TempDir()); err != nil {
		panic(fmt.Sprintf("could not set the %s environment variable", dryRunDir))
	}
	kubeadmPath := getKubeadmPath()
	kubeadmArgs := []string{"init", "--dry-run", "--ignore-preflight-errors=all"}
	kubeadmArgs = append(kubeadmArgs, args...)
	return RunCmd(kubeadmPath, kubeadmArgs...)
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
			_, _, _, err := runKubeadmInit(rt.args)
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
			args:     "--kubernetes-version=" + constants.CurrentKubernetesVersion.String(),
			expected: true,
		},
	}

	for _, rt := range initTest {
		t.Run(rt.name, func(t *testing.T) {
			_, _, _, err := runKubeadmInit(rt.args)
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
			name:     "can't load old v1alpha1 config",
			args:     "--config=testdata/init/v1alpha1.yaml",
			expected: false,
		},
		{
			name:     "can't load old v1alpha2 config",
			args:     "--config=testdata/init/v1alpha2.yaml",
			expected: false,
		},
		{
			name:     "can't load old v1alpha3 config",
			args:     "--config=testdata/init/v1alpha3.yaml",
			expected: false,
		},
		{
			name:     "can load v1beta1 config",
			args:     "--config=testdata/init/v1beta1.yaml",
			expected: true,
		},
		{
			name:     "don't allow mixed arguments v1beta1",
			args:     "--kubernetes-version=1.11.0 --config=testdata/init/v1beta1.yaml",
			expected: false,
		},
		{
			name:     "can load v1beta2 config",
			args:     "--config=testdata/init/v1beta2.yaml",
			expected: true,
		},
		{
			name:     "don't allow mixed arguments v1beta2",
			args:     "--kubernetes-version=1.11.0 --config=testdata/init/v1beta2.yaml",
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
			_, _, _, err := runKubeadmInit(rt.args)
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

func TestCmdInitCertPhaseCSR(t *testing.T) {
	tests := []struct {
		name          string
		baseName      string
		expectedError string
	}{
		{
			name:     "generate CSR",
			baseName: certs.KubeadmCertKubeletClient().BaseName,
		},
		{
			name:          "fails on CSR",
			baseName:      certs.KubeadmCertRootCA().BaseName,
			expectedError: "unknown flag: --csr-only",
		},
		{
			name:          "fails on all",
			baseName:      "all",
			expectedError: "unknown flag: --csr-only",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			csrDir := testutil.SetupTempDir(t)
			cert := certs.KubeadmCertKubeletClient()
			kubeadmPath := getKubeadmPath()
			_, stderr, _, err := RunCmd(kubeadmPath,
				"init",
				"phase",
				"certs",
				test.baseName,
				"--csr-only",
				"--csr-dir="+csrDir,
			)

			if test.expectedError != "" {
				cause := errors.Cause(err)
				_, ok := cause.(*exec.ExitError)
				if !ok {
					t.Fatalf("expected exitErr: got %T (%v)", cause, err)
				}

				if !strings.Contains(stderr, test.expectedError) {
					t.Errorf("expected %q to contain %q", stderr, test.expectedError)
				}
				return
			}

			if err != nil {
				t.Fatalf("couldn't run kubeadm: %v", err)
			}

			if _, _, err := pkiutil.TryLoadCSRAndKeyFromDisk(csrDir, cert.BaseName); err != nil {
				t.Fatalf("couldn't load certificate %q: %v", cert.BaseName, err)
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
			_, _, _, err := runKubeadmInit(rt.args)
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
			name: "feature gate IPv6DualStack=true",
			args: "--feature-gates=IPv6DualStack=true",
		},
		{
			name: "feature gate PublicKeysECDSA=true",
			args: "--feature-gates=PublicKeysECDSA=true",
		},
	}

	for _, rt := range initTest {
		t.Run(rt.name, func(t *testing.T) {
			_, _, exitcode, err := runKubeadmInit(rt.args)
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
