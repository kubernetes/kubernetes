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
	"testing"

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	certstestutil "k8s.io/kubernetes/cmd/kubeadm/app/util/certs"
	pkiutiltesting "k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil/testing"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

type testCertsData struct {
	testInitData
	cfg *kubeadmapi.InitConfiguration
}

func (t *testCertsData) Cfg() *kubeadmapi.InitConfiguration { return t.cfg }
func (t *testCertsData) ExternalCA() bool                   { return false }
func (t *testCertsData) CertificateDir() string             { return t.cfg.CertificatesDir }
func (t *testCertsData) CertificateWriteDir() string        { return t.cfg.CertificatesDir }

func TestCreateSparseCerts(t *testing.T) {
	for _, test := range certstestutil.GetSparseCertTestCases(t) {
		t.Run(test.Name, func(t *testing.T) {
			pkiutiltesting.Reset()

			tmpdir := testutil.SetupTempDir(t)
			defer os.RemoveAll(tmpdir)

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
