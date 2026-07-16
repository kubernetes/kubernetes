/*
Copyright 2018 The Kubernetes Authors.

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

package phases

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	certstestutil "k8s.io/kubernetes/cmd/kubeadm/app/util/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	pkiutiltesting "k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil/testing"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

type testCertsData struct {
	testInitData
	cfg *kubeadmapi.InitConfiguration
}

type testDryRunCertsData struct {
	testCertsData
	certificateDir      string
	certificateWriteDir string
}

func (t *testCertsData) Cfg() *kubeadmapi.InitConfiguration { return t.cfg }
func (t *testCertsData) ExternalCA() bool                   { return false }
func (t *testCertsData) CertificateDir() string             { return t.cfg.CertificatesDir }
func (t *testCertsData) CertificateWriteDir() string        { return t.cfg.CertificatesDir }
func (t *testDryRunCertsData) DryRun() bool                 { return true }
func (t *testDryRunCertsData) CertificateDir() string       { return t.certificateDir }
func (t *testDryRunCertsData) CertificateWriteDir() string  { return t.certificateWriteDir }

func TestCreateSparseCerts(t *testing.T) {
	for _, test := range certstestutil.GetSparseCertTestCases(t) {
		t.Run(test.Name, func(t *testing.T) {
			pkiutiltesting.Reset()

			tmpdir := t.TempDir()

			certstestutil.WritePKIFiles(t, tmpdir, test.Files)

			r := workflow.NewRunner()
			r.AppendPhase(NewCertsPhase())
			r.SetDataInitializer(func(*cobra.Command, []string) (workflow.RunData, error) {
				certsData := &testCertsData{
					cfg: testutil.GetDefaultInternalConfig(t),
				}
				certsData.cfg.CertificatesDir = tmpdir
				return certsData, nil
			})

			if err := r.Run([]string{}); (err != nil) != test.ExpectError {
				t.Fatalf("expected error to be %t, got %t (%v)", test.ExpectError, (err != nil), err)
			}
		})
	}
}

func TestRunCAPhaseCopiesExistingCAFilesToDryRunDir(t *testing.T) {
	for _, ca := range []*certsphase.KubeadmCert{
		certsphase.KubeadmCertRootCA(),
		certsphase.KubeadmCertFrontProxyCA(),
		certsphase.KubeadmCertEtcdCA(),
	} {
		t.Run(ca.Name, func(t *testing.T) {
			pkiutiltesting.Reset()

			sourceDir := t.TempDir()
			writeDir := t.TempDir()
			caCert, caKey := certstestutil.SetupCertificateAuthority(t)
			certPath, _ := pkiutil.PathsForCertAndKey(sourceDir, ca.BaseName)
			if err := os.MkdirAll(filepath.Dir(certPath), os.FileMode(0700)); err != nil {
				t.Fatalf("failed to create source directory for %s: %v", ca.BaseName, err)
			}
			if err := pkiutil.WriteCertAndKey(sourceDir, ca.BaseName, caCert, caKey); err != nil {
				t.Fatalf("failed to write source CA files for %s: %v", ca.BaseName, err)
			}

			cfg := testutil.GetDefaultInternalConfig(t)
			cfg.CertificatesDir = sourceDir
			data := &testDryRunCertsData{
				testCertsData:       testCertsData{cfg: cfg},
				certificateDir:      sourceDir,
				certificateWriteDir: writeDir,
			}

			if err := runCAPhase(ca)(data); err != nil {
				t.Fatalf("runCAPhase(%s) returned error: %v", ca.Name, err)
			}

			if _, err := pkiutil.TryLoadCertFromDisk(writeDir, ca.BaseName); err != nil {
				t.Fatalf("expected copied cert for %s in dry-run dir: %v", ca.BaseName, err)
			}
			if _, err := pkiutil.TryLoadKeyFromDisk(writeDir, ca.BaseName); err != nil {
				t.Fatalf("expected copied key for %s in dry-run dir: %v", ca.BaseName, err)
			}
		})
	}
}
