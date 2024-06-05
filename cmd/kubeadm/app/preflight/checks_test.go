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

package preflight

import (
	"bytes"
	"fmt"
	"net"
	"net/http"
	"os"
	"runtime"
	"strings"
	"testing"

	utiltesting "k8s.io/client-go/util/testing"

	"github.com/google/go-cmp/cmp"
	"github.com/lithammer/dedent"
	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

var (
	externalEtcdRootCAFileContent = dedent.Dedent(`
		-----BEGIN CERTIFICATE-----
		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
		Z9JkQayesM6D7uwbQJXd5rgy
		-----END CERTIFICATE-----
	`)

	externalEtcdCertFileContent = dedent.Dedent(`
		-----BEGIN CERTIFICATE-----
		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
		XRMVQjawUTknbAgHuE7/rEX8c27WUA==
		-----END CERTIFICATE-----
	`)
	externalEtcdKeyFileContent = dedent.Dedent(`
		-----BEGIN RSA PRIVATE KEY-----
		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
		-----END RSA PRIVATE KEY-----
	`)
)

type preflightCheckTest struct {
	msg string
}

func (pfct preflightCheckTest) Name() string {
	return "preflightCheckTest"
}

func (pfct preflightCheckTest) Check() (warning, errorList []error) {
	if pfct.msg == "warning" {
		return []error{errors.New("warning")}, nil
	}
	if pfct.msg != "" {
		return nil, []error{errors.New("fake error")}
	}
	return
}

func TestFileExistingCheck(t *testing.T) {
	f, err := os.CreateTemp("", "file-exist-check")
	if err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}
	defer utiltesting.CloseAndRemove(t, f)
	var tests = []struct {
		name          string
		check         FileExistingCheck
		expectedError bool
	}{
		{
			name: "File does not exist, so it's not available",
			check: FileExistingCheck{
				Path: "/does/not/exist",
			},
			expectedError: true,
		},
		{
			name: "File exists and is available",
			check: FileExistingCheck{
				Path: f.Name(),
			},
			expectedError: false,
		},
	}
	for _, rt := range tests {
		_, output := rt.check.Check()
		if (output != nil) != rt.expectedError {
			t.Errorf(
				"Failed FileExistingCheck:%v\n\texpectedError: %t\n\t  actual: %t",
				rt.name,
				rt.expectedError,
				(output != nil),
			)
		}
	}
}

func TestFileAvailableCheck(t *testing.T) {
	f, err := os.CreateTemp("", "file-avail-check")
	if err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}
	defer utiltesting.CloseAndRemove(t, f)
	var tests = []struct {
		name          string
		check         FileAvailableCheck
		expectedError bool
	}{
		{
			name: "The file does not exist",
			check: FileAvailableCheck{
				Path: "/does/not/exist",
			},
			expectedError: false,
		},
		{
			name: "The file exists",
			check: FileAvailableCheck{
				Path: f.Name(),
			},
			expectedError: true,
		},
	}
	for _, rt := range tests {
		_, output := rt.check.Check()
		if (output != nil) != rt.expectedError {
			t.Errorf(
				"Failed FileAvailableCheck:%v\n\texpectedError: %t\n\t  actual: %t",
				rt.name,
				rt.expectedError,
				(output != nil),
			)
		}
	}
}

func TestFileContentCheck(t *testing.T) {
	f, err := os.CreateTemp("", "file-content-check")
	if err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}
	defer os.Remove(f.Name())
	var tests = []struct {
		name          string
		check         FileContentCheck
		expectedError bool
	}{
		{
			name: "File exists and has matching content",
			check: FileContentCheck{
				Path:    f.Name(),
				Content: []byte("Test FileContentCheck"),
			},
			expectedError: false,
		},
		{
			name: "File exists, content is nil",
			check: FileContentCheck{
				Path:    f.Name(),
				Content: nil,
			},
			expectedError: false,
		},
		{
			name: "File exists but has unexpected content",
			check: FileContentCheck{
				Path:    f.Name(),
				Content: []byte("foo"),
			},
			expectedError: true,
		},
		{
			name: "File does not exist, content is not nil",
			check: FileContentCheck{
				Path:    "/does/not/exist",
				Content: []byte("foo"),
			},
			expectedError: true,
		},
		{
			name: "File dose not exist, content is nil",
			check: FileContentCheck{
				Path:    "/does/not/exist",
				Content: nil,
			},
			expectedError: true,
		},
	}
	if _, err = f.WriteString("Test FileContentCheck"); err != nil {
		t.Fatalf("Failed to write to file: %v", err)
	}
	for _, rt := range tests {
		_, output := rt.check.Check()
		if (len(output) > 0) != rt.expectedError {
			t.Errorf(
				"Failed FileContentCheck:%v\n\texpectedError: %t\n\t  actual: %t",
				rt.name,
				rt.expectedError,
				(len(output) > 0),
			)
		}
	}
}

func TestDirAvailableCheck(t *testing.T) {
	fileDir, err := os.MkdirTemp("", "dir-avail-check")
	if err != nil {
		t.Fatalf("failed creating directory: %v", err)
	}
	defer os.RemoveAll(fileDir)
	var tests = []struct {
		name          string
		check         DirAvailableCheck
		expectedError bool
	}{
		{
			name: "Directory exists and is empty",
			check: DirAvailableCheck{
				Path: fileDir,
			},
			expectedError: false,
		},
		{
			name: "Directory exists and has something",
			check: DirAvailableCheck{
				Path: os.TempDir(), // a directory was created previously in this test
			},
			expectedError: true,
		},
		{
			name: "Directory does not exist",
			check: DirAvailableCheck{
				Path: "/does/not/exist",
			},
			expectedError: false,
		},
	}
	for _, rt := range tests {
		_, output := rt.check.Check()
		if (output != nil) != rt.expectedError {
			t.Errorf(
				"Failed DirAvailableCheck:%v\n\texpectedError: %t\n\t  actual: %t",
				rt.name,
				rt.expectedError,
				(output != nil),
			)
		}
	}
}

func TestPortOpenCheck(t *testing.T) {
	ln, err := net.Listen("tcp", ":0")
	if err != nil {
		t.Fatalf("could not listen on local network: %v", err)
	}
	defer ln.Close()
	var tests = []struct {
		name          string
		check         PortOpenCheck
		expectedError bool
	}{
		{
			name:          "Port is available",
			check:         PortOpenCheck{port: 0},
			expectedError: false,
		},
		{
			name:          "Port is not available",
			check:         PortOpenCheck{port: ln.Addr().(*net.TCPAddr).Port},
			expectedError: true,
		},
	}
	for _, rt := range tests {
		_, output := rt.check.Check()
		if (output != nil) != rt.expectedError {
			t.Errorf(
				"Failed PortOpenCheck:%v\n\texpectedError: %t\n\t  actual: %t",
				rt.name,
				rt.expectedError,
				(output != nil),
			)
		}
	}
}

func TestRunChecks(t *testing.T) {
	var tokenTest = []struct {
		p        []Checker
		expected bool
		output   string
	}{
		{[]Checker{}, true, ""},
		{[]Checker{preflightCheckTest{"warning"}}, true, "\t[WARNING preflightCheckTest]: warning\n"}, // should just print warning
		{[]Checker{preflightCheckTest{"error"}}, false, ""},
		{[]Checker{preflightCheckTest{"test"}}, false, ""},
		{[]Checker{DirAvailableCheck{Path: "/does/not/exist"}}, true, ""},
		{[]Checker{DirAvailableCheck{Path: "/"}}, false, ""},
		{[]Checker{FileAvailableCheck{Path: "/does/not/exist"}}, true, ""},
		{[]Checker{FileContentCheck{Path: "/does/not/exist"}}, false, ""},
		{[]Checker{FileContentCheck{Path: "/"}}, true, ""},
		{[]Checker{FileContentCheck{Path: "/", Content: []byte("does not exist")}}, false, ""},
		{[]Checker{InPathCheck{executable: "foobarbaz", exec: exec.New()}}, true, "\t[WARNING FileExisting-foobarbaz]: foobarbaz not found in system path\n"},
		{[]Checker{InPathCheck{executable: "foobarbaz", mandatory: true, exec: exec.New()}}, false, ""},
		{[]Checker{InPathCheck{executable: "foobar", mandatory: false, exec: exec.New(), suggestion: "install foobar"}}, true, "\t[WARNING FileExisting-foobar]: foobar not found in system path\nSuggestion: install foobar\n"},
	}
	for _, rt := range tokenTest {
		buf := new(bytes.Buffer)
		actual := RunChecks(rt.p, buf, sets.New[string]())
		if (actual == nil) != rt.expected {
			t.Errorf(
				"failed RunChecks:\n\texpected: %t\n\t  actual: %t",
				rt.expected,
				(actual == nil),
			)
		}
		if buf.String() != rt.output {
			t.Errorf(
				"failed RunChecks:\n\texpected: %s\n\t  actual: %s",
				rt.output,
				buf.String(),
			)
		}
	}
}
func TestConfigRootCAs(t *testing.T) {
	f, err := os.CreateTemp(os.TempDir(), "kubeadm-external-etcd-test-cafile")
	if err != nil {
		t.Errorf("failed configRootCAs:\n\texpected: succeed creating temp CA file\n\tactual:%v", err)
	}
	defer utiltesting.CloseAndRemove(t, f)
	if _, err := f.Write([]byte(externalEtcdRootCAFileContent)); err != nil {
		t.Errorf("failed configRootCAs:\n\texpected: succeed writing contents to temp CA file %s\n\tactual:%v", f.Name(), err)
	}

	c := ExternalEtcdVersionCheck{Etcd: kubeadmapi.Etcd{External: &kubeadmapi.ExternalEtcd{CAFile: f.Name()}}}

	config, err := c.configRootCAs(nil)
	if err != nil {
		t.Errorf(
			"failed configRootCAs:\n\texpected: has no error\n\tactual:%v",
			err,
		)
	}
	if config.RootCAs == nil {
		t.Errorf(
			"failed configRootCAs:\n\texpected: RootCAs not equal to nil\n\tactual:%v",
			config.RootCAs,
		)
	}
}
func TestConfigCertAndKey(t *testing.T) {
	certFile, err := os.CreateTemp(os.TempDir(), "kubeadm-external-etcd-test-certfile")
	if err != nil {
		t.Errorf(
			"failed configCertAndKey:\n\texpected: succeed creating temp CertFile file\n\tactual:%v",
			err,
		)
	}
	defer utiltesting.CloseAndRemove(t, certFile)
	if _, err := certFile.Write([]byte(externalEtcdCertFileContent)); err != nil {
		t.Errorf(
			"failed configCertAndKey:\n\texpected: succeed writing contents to temp CertFile file %s\n\tactual:%v",
			certFile.Name(),
			err,
		)
	}

	keyFile, err := os.CreateTemp(os.TempDir(), "kubeadm-external-etcd-test-keyfile")
	if err != nil {
		t.Errorf(
			"failed configCertAndKey:\n\texpected: succeed creating temp KeyFile file\n\tactual:%v",
			err,
		)
	}
	defer utiltesting.CloseAndRemove(t, keyFile)
	if _, err := keyFile.Write([]byte(externalEtcdKeyFileContent)); err != nil {
		t.Errorf(
			"failed configCertAndKey:\n\texpected: succeed writing contents to temp KeyFile file %s\n\tactual:%v",
			keyFile.Name(),
			err,
		)
	}
	c := ExternalEtcdVersionCheck{
		Etcd: kubeadmapi.Etcd{
			External: &kubeadmapi.ExternalEtcd{
				CertFile: certFile.Name(),
				KeyFile:  keyFile.Name(),
			},
		},
	}

	config, err := c.configCertAndKey(nil)
	if err != nil {
		t.Errorf(
			"failed configCertAndKey:\n\texpected: has no error\n\tactual:%v",
			err,
		)
	}
	if config.Certificates == nil {
		t.Errorf(
			"failed configCertAndKey:\n\texpected: Certificates not equal to nil\n\tactual:%v",
			config.Certificates,
		)
	}
}

func TestKubernetesVersionCheck(t *testing.T) {
	var tests = []struct {
		check          KubernetesVersionCheck
		expectWarnings bool
	}{
		{
			check: KubernetesVersionCheck{
				KubeadmVersion:    "v1.6.6", //Same version
				KubernetesVersion: "v1.6.6",
			},
			expectWarnings: false,
		},
		{
			check: KubernetesVersionCheck{
				KubeadmVersion:    "v1.6.6", //KubernetesVersion version older than KubeadmVersion
				KubernetesVersion: "v1.5.5",
			},
			expectWarnings: false,
		},
		{
			check: KubernetesVersionCheck{
				KubeadmVersion:    "v1.6.6", //KubernetesVersion newer than KubeadmVersion, within the same minor release (new patch)
				KubernetesVersion: "v1.6.7",
			},
			expectWarnings: false,
		},
		{
			check: KubernetesVersionCheck{
				KubeadmVersion:    "v1.6.6", //KubernetesVersion newer than KubeadmVersion, in a different minor/in pre-release
				KubernetesVersion: "v1.7.0-alpha.0",
			},
			expectWarnings: true,
		},
		{
			check: KubernetesVersionCheck{
				KubeadmVersion:    "v1.6.6", //KubernetesVersion newer than KubeadmVersion, in a different minor/stable
				KubernetesVersion: "v1.7.0",
			},
			expectWarnings: true,
		},
		{
			check: KubernetesVersionCheck{
				KubeadmVersion:    "v0.0.0", //"super-custom" builds - Skip this check
				KubernetesVersion: "v1.7.0",
			},
			expectWarnings: false,
		},
	}

	for _, rt := range tests {
		warning, _ := rt.check.Check()
		if (warning != nil) != rt.expectWarnings {
			t.Errorf(
				"failed KubernetesVersionCheck:\n\texpected: %t\n\t  actual: %t (KubeadmVersion:%s, KubernetesVersion: %s)",
				rt.expectWarnings,
				(warning != nil),
				rt.check.KubeadmVersion,
				rt.check.KubernetesVersion,
			)
		}
	}
}

func TestHTTPProxyCIDRCheck(t *testing.T) {
	var tests = []struct {
		check          HTTPProxyCIDRCheck
		expectWarnings bool
	}{
		{
			check: HTTPProxyCIDRCheck{
				Proto: "https",
				CIDR:  "127.0.0.0/8",
			}, // Loopback addresses never should produce proxy warnings
			expectWarnings: false,
		},
		{
			check: HTTPProxyCIDRCheck{
				Proto: "https",
				CIDR:  "10.96.0.0/12",
			}, // Expected to be accessed directly, we set NO_PROXY to 10.0.0.0/8
			expectWarnings: false,
		},
		{
			check: HTTPProxyCIDRCheck{
				Proto: "https",
				CIDR:  "192.168.0.0/16",
			}, // Expected to go via proxy as this range is not listed in NO_PROXY
			expectWarnings: true,
		},
		{
			check: HTTPProxyCIDRCheck{
				Proto: "https",
				CIDR:  "2001:db8::/56",
			}, // Expected to be accessed directly, part of 2001:db8::/48 in NO_PROXY
			expectWarnings: false,
		},
		{
			check: HTTPProxyCIDRCheck{
				Proto: "https",
				CIDR:  "2001:db8:1::/56",
			}, // Expected to go via proxy, range is not in 2001:db8::/48
			expectWarnings: true,
		},
	}

	resetProxyEnv(t)

	for _, rt := range tests {
		warning, _ := rt.check.Check()
		if (warning != nil) != rt.expectWarnings {
			t.Errorf(
				"failed HTTPProxyCIDRCheck:\n\texpected: %t\n\t  actual: %t (CIDR:%s). Warnings: %v",
				rt.expectWarnings,
				(warning != nil),
				rt.check.CIDR,
				warning,
			)
		}
	}
}

func TestHTTPProxyCheck(t *testing.T) {
	var tests = []struct {
		name           string
		check          HTTPProxyCheck
		expectWarnings bool
	}{
		{
			name: "Loopback address",
			check: HTTPProxyCheck{
				Proto: "https",
				Host:  "127.0.0.1",
			}, // Loopback addresses never should produce proxy warnings
			expectWarnings: false,
		},
		{
			name: "IPv4 direct access",
			check: HTTPProxyCheck{
				Proto: "https",
				Host:  "10.96.0.1",
			}, // Expected to be accessed directly, we set NO_PROXY to 10.0.0.0/8
			expectWarnings: false,
		},
		{
			name: "IPv4 via proxy",
			check: HTTPProxyCheck{
				Proto: "https",
				Host:  "192.168.0.1",
			}, // Expected to go via proxy as this range is not listed in NO_PROXY
			expectWarnings: true,
		},
		{
			name: "IPv6 direct access",
			check: HTTPProxyCheck{
				Proto: "https",
				Host:  "[2001:db8::1:15]",
			}, // Expected to be accessed directly, part of 2001:db8::/48 in NO_PROXY
			expectWarnings: false,
		},
		{
			name: "IPv6 via proxy",
			check: HTTPProxyCheck{
				Proto: "https",
				Host:  "[2001:db8:1::1:15]",
			}, // Expected to go via proxy, range is not in 2001:db8::/48
			expectWarnings: true,
		},
		{
			name: "IPv6 direct access, no brackets",
			check: HTTPProxyCheck{
				Proto: "https",
				Host:  "2001:db8::1:15",
			}, // Expected to be accessed directly, part of 2001:db8::/48 in NO_PROXY
			expectWarnings: false,
		},
		{
			name: "IPv6 via proxy, no brackets",
			check: HTTPProxyCheck{
				Proto: "https",
				Host:  "2001:db8:1::1:15",
			}, // Expected to go via proxy, range is not in 2001:db8::/48
			expectWarnings: true,
		},
	}

	resetProxyEnv(t)

	for _, rt := range tests {
		warning, _ := rt.check.Check()
		if (warning != nil) != rt.expectWarnings {
			t.Errorf(
				"%s failed HTTPProxyCheck:\n\texpected: %t\n\t  actual: %t (Host:%s). Warnings: %v",
				rt.name,
				rt.expectWarnings,
				(warning != nil),
				rt.check.Host,
				warning,
			)
		}
	}
}

// resetProxyEnv is helper function that unsets all *_proxy variables.
func resetProxyEnv(t *testing.T) {
	for _, e := range os.Environ() {
		key, value, _ := strings.Cut(e, "=")
		if strings.HasSuffix(strings.ToLower(key), "_proxy") {
			t.Cleanup(func() { os.Setenv(key, value) })
			os.Unsetenv(key)
		}
	}

	t.Setenv("HTTP_PROXY", "http://proxy.example.com:3128")
	t.Setenv("HTTPS_PROXY", "https://proxy.example.com:3128")
	t.Setenv("NO_PROXY", "example.com,10.0.0.0/8,2001:db8::/48")
	// Check if we can reliably execute tests:
	// ProxyFromEnvironment caches the *_proxy environment variables and
	// if ProxyFromEnvironment already executed before our test with empty
	// HTTP_PROXY it will make these tests return false positive failures
	req, err := http.NewRequest("GET", "http://host.fake.tld/", nil)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	proxy, err := http.ProxyFromEnvironment(req)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if proxy == nil {
		t.Skip("test skipped as ProxyFromEnvironment already initialized in environment without defined HTTP proxy")
	}
	t.Log("http.ProxyFromEnvironment is usable, continue executing test")
}

func TestKubeletVersionCheck(t *testing.T) {
	minimumKubeletVersion := version.MustParseSemantic("v1.3.0")
	minimumControlPlaneVersion := version.MustParseSemantic("v1.3.0")
	currentKubernetesVersion := version.MustParseSemantic("v1.4.0")
	cases := []struct {
		kubeletVersion string
		k8sVersion     string
		expectErrors   bool
		expectWarnings bool
	}{
		{"v" + currentKubernetesVersion.WithPatch(2).String(), "", false, false},                                                           // check minimally supported version when there is no information about control plane
		{"v1.1.0", "v1.11.8", true, false},                                                                                                 // too old kubelet, should fail.
		{"v" + minimumKubeletVersion.String(), minimumControlPlaneVersion.WithPatch(5).String(), false, false},                             // kubelet within same major.minor as control plane
		{"v" + minimumKubeletVersion.WithPatch(5).String(), minimumControlPlaneVersion.WithPatch(1).String(), false, false},                // kubelet is newer, but still within same major.minor as control plane
		{"v" + minimumKubeletVersion.String(), currentKubernetesVersion.WithPatch(1).String(), false, false},                               // kubelet is lower than control plane, but newer than minimally supported
		{"v" + currentKubernetesVersion.WithPreRelease("alpha.1").String(), minimumControlPlaneVersion.WithPatch(1).String(), true, false}, // kubelet is newer (development build) than control plane, should fail.
		{"v" + currentKubernetesVersion.String(), minimumControlPlaneVersion.WithPatch(5).String(), true, false},                           // kubelet is newer (release) than control plane, should fail.
	}

	for _, tc := range cases {
		t.Run(tc.kubeletVersion, func(t *testing.T) {
			fcmd := fakeexec.FakeCmd{
				OutputScript: []fakeexec.FakeAction{
					func() ([]byte, []byte, error) { return []byte("Kubernetes " + tc.kubeletVersion), nil, nil },
				},
			}
			fexec := &fakeexec.FakeExec{
				CommandScript: []fakeexec.FakeCommandAction{
					func(cmd string, args ...string) exec.Cmd { return fakeexec.InitFakeCmd(&fcmd, cmd, args...) },
				},
			}

			check := KubeletVersionCheck{KubernetesVersion: tc.k8sVersion, exec: fexec, minKubeletVersion: minimumKubeletVersion}
			warnings, errors := check.Check()

			switch {
			case warnings != nil && !tc.expectWarnings:
				t.Errorf("KubeletVersionCheck: unexpected warnings for kubelet version %q and Kubernetes version %q. Warnings: %v", tc.kubeletVersion, tc.k8sVersion, warnings)
			case warnings == nil && tc.expectWarnings:
				t.Errorf("KubeletVersionCheck: expected warnings for kubelet version %q and Kubernetes version %q but got nothing", tc.kubeletVersion, tc.k8sVersion)
			case errors != nil && !tc.expectErrors:
				t.Errorf("KubeletVersionCheck: unexpected errors for kubelet version %q and Kubernetes version %q. errors: %v", tc.kubeletVersion, tc.k8sVersion, errors)
			case errors == nil && tc.expectErrors:
				t.Errorf("KubeletVersionCheck: expected errors for kubelet version %q and Kubernetes version %q but got nothing", tc.kubeletVersion, tc.k8sVersion)
			}
		})
	}
}

func TestSetHasItemOrAll(t *testing.T) {
	var tests = []struct {
		ignoreSet      sets.Set[string]
		testString     string
		expectedResult bool
	}{
		{sets.New[string](), "foo", false},
		{sets.New("all"), "foo", true},
		{sets.New("all", "bar"), "foo", true},
		{sets.New("bar"), "foo", false},
		{sets.New("baz", "foo", "bar"), "foo", true},
		{sets.New("baz", "bar", "foo"), "Foo", true},
	}

	for _, rt := range tests {
		t.Run(rt.testString, func(t *testing.T) {
			result := setHasItemOrAll(rt.ignoreSet, rt.testString)
			if result != rt.expectedResult {
				t.Errorf(
					"setHasItemOrAll: expected: %v actual: %v (arguments: %q, %q)",
					rt.expectedResult, result,
					rt.ignoreSet,
					rt.testString,
				)
			}
		})
	}
}

func TestNumCPUCheck(t *testing.T) {
	var tests = []struct {
		numCPU      int
		numErrors   int
		numWarnings int
	}{
		{0, 0, 0},
		{999999999, 1, 0},
	}

	for _, rt := range tests {
		t.Run(fmt.Sprintf("number of CPUs: %d", rt.numCPU), func(t *testing.T) {
			warnings, errors := NumCPUCheck{NumCPU: rt.numCPU}.Check()
			if len(warnings) != rt.numWarnings {
				t.Errorf("expected %d warning(s) but got %d: %q", rt.numWarnings, len(warnings), warnings)
			}
			if len(errors) != rt.numErrors {
				t.Errorf("expected %d warning(s) but got %d: %q", rt.numErrors, len(errors), errors)
			}
		})
	}
}

func TestMemCheck(t *testing.T) {
	// skip this test, if OS in not Linux, since it will ONLY pass on Linux.
	if runtime.GOOS != "linux" {
		t.Skip("unsupported OS for memory check test ")
	}

	var tests = []struct {
		minimum        uint64
		expectedErrors int
	}{
		{0, 0},
		{9999999999999999, 1},
	}

	for _, rt := range tests {
		t.Run(fmt.Sprintf("MemoryCheck{%d}", rt.minimum), func(t *testing.T) {
			warnings, errors := MemCheck{Mem: rt.minimum}.Check()
			if len(warnings) > 0 {
				t.Errorf("expected 0 warnings but got %d: %q", len(warnings), warnings)
			} else if len(errors) != rt.expectedErrors {
				t.Errorf("expected %d error(s) but got %d: %q", rt.expectedErrors, len(errors), errors)
			}
		})
	}
}

func TestInitIPCheck(t *testing.T) {
	// skip this test, if OS in not Linux, since it will ONLY pass on Linux.
	if runtime.GOOS != "linux" {
		t.Skip("unsupported OS")
	}
	// should be a privileged user for the `init` command, otherwise just skip it.
	isPrivileged := IsPrivilegedUserCheck{}
	if _, err := isPrivileged.Check(); err != nil {
		t.Skip("not a privileged user")
	}
	internalcfg, err := configutil.DefaultedStaticInitConfiguration()
	if err != nil {
		t.Fatalf("unexpected failure when defaulting InitConfiguration: %v", err)
	}
	internalcfg.LocalAPIEndpoint.AdvertiseAddress = "" // AdvertiseAddress is optional, it could be auto-detected.
	ipv4File := "FileContent--proc-sys-net-ipv4-ip_forward"
	ipv6File := "FileContent--proc-sys-net-ipv6-conf-default-forwarding"
	var tests = []struct {
		testName    string
		PodSubnet   string
		serviceCidr string
		expStr      []string
	}{
		{
			testName:    "dual stack",
			PodSubnet:   "fda9:d324:354d:0::/56",
			serviceCidr: "10.96.0.0/16,fda9:d324:354d:1::/112",
			expStr:      []string{"FileContent--proc-sys-net-ipv4-ip_forward", "FileContent--proc-sys-net-ipv6-conf-default-forwarding"},
		},
		{
			testName:    "single stack ipv4",
			PodSubnet:   "10.244.0.0/16",
			serviceCidr: "10.96.0.0/16",
			expStr:      []string{"FileContent--proc-sys-net-ipv4-ip_forward"},
		},
		{
			testName:    "single stack ipv6",
			PodSubnet:   "fda9:d324:354d:0::/56",
			serviceCidr: "fda9:d324:354d:1::/112",
			expStr:      []string{"FileContent--proc-sys-net-ipv6-conf-default-forwarding"},
		},
	}

	for _, rt := range tests {
		t.Run(rt.testName, func(t *testing.T) {
			checkList := []string{}
			internalcfg.Networking.ServiceSubnet = rt.serviceCidr
			internalcfg.Networking.PodSubnet = rt.PodSubnet
			checks, err := InitNodeChecks(exec.New(), internalcfg, nil, false, false)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			for _, check := range checks {
				if check.Name() == ipv4File {
					checkList = append(checkList, ipv4File)
				}
				if check.Name() == ipv6File {
					checkList = append(checkList, ipv6File)
				}
			}
			if diff := cmp.Diff(checkList, rt.expStr); diff != "" {
				t.Fatalf("unexpected file content check (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestJoinIPCheck(t *testing.T) {
	// skip this test, if OS in not Linux, since it will ONLY pass on Linux.
	if runtime.GOOS != "linux" {
		t.Skip("unsupported OS")
	}
	// should be a privileged user for the `join` command, otherwise just skip it.
	isPrivileged := IsPrivilegedUserCheck{}
	if _, err := isPrivileged.Check(); err != nil {
		t.Skip("not a privileged user")
	}

	opts := configutil.LoadOrDefaultConfigurationOptions{
		SkipCRIDetect: true,
	}

	internalcfg, err := configutil.DefaultedJoinConfiguration(&kubeadmapiv1.JoinConfiguration{
		Discovery: kubeadmapiv1.Discovery{
			BootstrapToken: &kubeadmapiv1.BootstrapTokenDiscovery{
				Token:                    configutil.PlaceholderToken.Token.String(),
				APIServerEndpoint:        "kube-apiserver:6443",
				UnsafeSkipCAVerification: true,
			},
		},
	}, opts)
	if err != nil {
		t.Fatalf("unexpected failure when defaulting JoinConfiguration: %v", err)
	}
	ipv4File := "FileContent--proc-sys-net-ipv4-ip_forward"
	ipv6File := "FileContent--proc-sys-net-ipv6-conf-default-forwarding"
	var tests = []struct {
		testName string
		endpoint string
		expStr   []string
	}{
		{
			testName: "single stack ipv4",
			endpoint: "10.244.0.0:1234",
			expStr:   []string{"FileContent--proc-sys-net-ipv4-ip_forward"},
		},
		{
			testName: "single stack ipv6",
			endpoint: "[fda9:d324:354d:0::]:1234",
			expStr:   []string{"FileContent--proc-sys-net-ipv6-conf-default-forwarding"},
		},
	}

	for _, rt := range tests {
		t.Run(rt.testName, func(t *testing.T) {
			checkList := []string{}
			internalcfg.Discovery.BootstrapToken.APIServerEndpoint = rt.endpoint
			checks, err := JoinNodeChecks(exec.New(), internalcfg, nil)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			for _, check := range checks {
				if check.Name() == ipv4File {
					checkList = append(checkList, ipv4File)
				}
				if check.Name() == ipv6File {
					checkList = append(checkList, ipv6File)
				}
			}
			if diff := cmp.Diff(checkList, rt.expStr); diff != "" {
				t.Fatalf("unexpected file content check (-want,+got):\n%s", diff)
			}
		})
	}
}
