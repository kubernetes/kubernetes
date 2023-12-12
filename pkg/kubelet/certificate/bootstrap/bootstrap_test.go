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

package bootstrap

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	utiltesting "k8s.io/client-go/util/testing"

	"github.com/google/go-cmp/cmp"
	certificatesv1 "k8s.io/api/certificates/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/fake"
	certificatesclient "k8s.io/client-go/kubernetes/typed/certificates/v1beta1"
	restclient "k8s.io/client-go/rest"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/util/certificate"
	"k8s.io/client-go/util/keyutil"
)

func copyFile(src, dst string) (err error) {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer func() {
		cerr := out.Close()
		if err == nil {
			err = cerr
		}
	}()
	_, err = io.Copy(out, in)
	return err
}

func TestLoadClientConfig(t *testing.T) {
	//Create a temporary folder under tmp to store the required certificate files and configuration files.
	fileDir := t.TempDir()
	//Copy the required certificate file to the temporary directory.
	copyFile("./testdata/mycertinvalid.crt", fileDir+"/mycertinvalid.crt")
	copyFile("./testdata/mycertvalid.crt", fileDir+"/mycertvalid.crt")
	copyFile("./testdata/mycertinvalid.key", fileDir+"/mycertinvalid.key")
	copyFile("./testdata/mycertvalid.key", fileDir+"/mycertvalid.key")
	testDataValid := []byte(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority: ca-a.crt
    server: https://cluster-a.com
  name: cluster-a
- cluster:
    server: https://cluster-b.com
  name: cluster-b
contexts:
- context:
    cluster: cluster-a
    namespace: ns-a
    user: user-a
  name: context-a
- context:
    cluster: cluster-b
    namespace: ns-b
    user: user-b
  name: context-b
current-context: context-b
users:
- name: user-a
  user:
    client-certificate: mycertvalid.crt
    client-key: mycertvalid.key
- name: user-b
  user:
    client-certificate: mycertvalid.crt
    client-key: mycertvalid.key

`)
	filevalid, err := os.CreateTemp(fileDir, "kubeconfigvalid")
	if err != nil {
		t.Fatal(err)
	}
	// os.CreateTemp also opens the file, and removing it without closing it will result in a failure.
	defer filevalid.Close()
	os.WriteFile(filevalid.Name(), testDataValid, os.FileMode(0755))

	testDataInvalid := []byte(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority: ca-a.crt
    server: https://cluster-a.com
  name: cluster-a
- cluster:
    server: https://cluster-b.com
  name: cluster-b
contexts:
- context:
    cluster: cluster-a
    namespace: ns-a
    user: user-a
  name: context-a
- context:
    cluster: cluster-b
    namespace: ns-b
    user: user-b
  name: context-b
current-context: context-b
users:
- name: user-a
  user:
    client-certificate: mycertinvalid.crt
    client-key: mycertinvalid.key
- name: user-b
  user:
    client-certificate: mycertinvalid.crt
    client-key: mycertinvalid.key

`)
	fileinvalid, err := os.CreateTemp(fileDir, "kubeconfiginvalid")
	if err != nil {
		t.Fatal(err)
	}
	defer fileinvalid.Close()
	os.WriteFile(fileinvalid.Name(), testDataInvalid, os.FileMode(0755))

	testDatabootstrap := []byte(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority: ca-a.crt
    server: https://cluster-a.com
  name: cluster-a
- cluster:
    server: https://cluster-b.com
  name: cluster-b
contexts:
- context:
    cluster: cluster-a
    namespace: ns-a
    user: user-a
  name: context-a
- context:
    cluster: cluster-b
    namespace: ns-b
    user: user-b
  name: context-b
current-context: context-b
users:
- name: user-a
  user:
   token: mytoken-b
- name: user-b
  user:
   token: mytoken-b
`)
	fileboot, err := os.CreateTemp(fileDir, "kubeconfig")
	if err != nil {
		t.Fatal(err)
	}
	defer fileboot.Close()
	os.WriteFile(fileboot.Name(), testDatabootstrap, os.FileMode(0755))

	dir, err := os.MkdirTemp(fileDir, "k8s-test-certstore-current")
	if err != nil {
		t.Fatalf("Unable to create the test directory %q: %v", dir, err)
	}

	store, err := certificate.NewFileStore("kubelet-client", dir, dir, "", "")
	if err != nil {
		t.Errorf("unable to build bootstrap cert store")
	}

	tests := []struct {
		name                 string
		kubeconfigPath       string
		bootstrapPath        string
		certDir              string
		expectedCertConfig   *restclient.Config
		expectedClientConfig *restclient.Config
	}{
		{
			name:           "bootstrapPath is empty",
			kubeconfigPath: filevalid.Name(),
			bootstrapPath:  "",
			certDir:        dir,
			expectedCertConfig: &restclient.Config{
				Host: "https://cluster-b.com",
				TLSClientConfig: restclient.TLSClientConfig{
					CertFile: filepath.Join(fileDir, "mycertvalid.crt"),
					KeyFile:  filepath.Join(fileDir, "mycertvalid.key"),
				},
				BearerToken: "",
			},
			expectedClientConfig: &restclient.Config{
				Host: "https://cluster-b.com",
				TLSClientConfig: restclient.TLSClientConfig{
					CertFile: filepath.Join(fileDir, "mycertvalid.crt"),
					KeyFile:  filepath.Join(fileDir, "mycertvalid.key"),
				},
				BearerToken: "",
			},
		},
		{
			name:           "bootstrap path is set and the contents of kubeconfigPath are valid",
			kubeconfigPath: filevalid.Name(),
			bootstrapPath:  fileboot.Name(),
			certDir:        dir,
			expectedCertConfig: &restclient.Config{
				Host: "https://cluster-b.com",
				TLSClientConfig: restclient.TLSClientConfig{
					CertFile: filepath.Join(fileDir, "mycertvalid.crt"),
					KeyFile:  filepath.Join(fileDir, "mycertvalid.key"),
				},
				BearerToken: "",
			},
			expectedClientConfig: &restclient.Config{
				Host: "https://cluster-b.com",
				TLSClientConfig: restclient.TLSClientConfig{
					CertFile: filepath.Join(fileDir, "mycertvalid.crt"),
					KeyFile:  filepath.Join(fileDir, "mycertvalid.key"),
				},
				BearerToken: "",
			},
		},
		{
			name:           "bootstrap path is set and the contents of kubeconfigPath are not valid",
			kubeconfigPath: fileinvalid.Name(),
			bootstrapPath:  fileboot.Name(),
			certDir:        dir,
			expectedCertConfig: &restclient.Config{
				Host:            "https://cluster-b.com",
				TLSClientConfig: restclient.TLSClientConfig{},
				BearerToken:     "mytoken-b",
			},
			expectedClientConfig: &restclient.Config{
				Host: "https://cluster-b.com",
				TLSClientConfig: restclient.TLSClientConfig{
					CertFile: store.CurrentPath(),
					KeyFile:  store.CurrentPath(),
				},
				BearerToken: "",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			certConfig, clientConfig, err := LoadClientConfig(test.kubeconfigPath, test.bootstrapPath, test.certDir)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(certConfig, test.expectedCertConfig) {
				t.Errorf("Unexpected certConfig: %s", cmp.Diff(certConfig, test.expectedCertConfig))
			}
			if !reflect.DeepEqual(clientConfig, test.expectedClientConfig) {
				t.Errorf("Unexpected clientConfig: %s", cmp.Diff(clientConfig, test.expectedClientConfig))
			}
		})
	}
}

func TestLoadRESTClientConfig(t *testing.T) {
	file, err := os.CreateTemp("", "my.ca")
	if err != nil {
		t.Fatalf("could not create tempfile: %v", err)
	}
	defer utiltesting.CloseAndRemove(t, file)

	caCert := `-----BEGIN CERTIFICATE-----
MIICyDCCAbCgAwIBAgIBADANBgkqhkiG9w0BAQsFADAVMRMwEQYDVQQDEwprdWJl
cm5ldGVzMB4XDTE5MTEyMDAwNDk0MloXDTI5MTExNzAwNDk0MlowFTETMBEGA1UE
AxMKa3ViZXJuZXRlczCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMqQ
ctECzA8yFSuVYupOUYgrTmfQeKe/9BaDWagaq7ow9+I2IvsfWFvlrD8QQr8sea6q
xjq7TV67Vb4RxBaoYDA+yI5vIcujWUxULun64lu3Q6iC1sj2UnmUpIdgazRXXEkZ
vxA6EbAnoxA0+lBOn1CZWl23IQ4s70o2hZ7wIp/vevB88RRRjqtvgc5elsjsbmDF
LS7L1Zuye8c6gS93bR+VjVmSIfr1IEq0748tIIyXjAVCWPVCvuP41MlfPc/JVpZD
uD2+pO6ZYREcdAnOf2eD4/eLOMKko4L1dSFy9JKM5PLnOC0Zk0AYOd1vS8DTAfxj
XPEIY8OBYFhlsxf4TE8CAwEAAaMjMCEwDgYDVR0PAQH/BAQDAgKkMA8GA1UdEwEB
/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADggEBAH/OYq8zyl1+zSTmuow3yI/15PL1
dl8hB7IKnZNWmC/LTdm/+noh3Sb1IdRv6HkKg/GUn0UMuRUngLhju3EO4ozJPQcX
quaxzgmTKNWJ6ErDvRvWhGX0ZcbdBfZv+dowyRqzd5nlJ49hC+NrtFFQq6P05BYn
7SemguqeXmXwIj2Sa+1DeR6lRm9o8shAYjnyThUFqaMn18kI3SANJ5vk/3DFrPEO
CKC9EzFku2kuxg2dM12PbRGZQ2o0K6HEZgrrIKTPOy3ocb8r9M0aSFhjOV/NqGA4
SaupXSW6XfvIi/UHoIbU3pNcsnUJGnQfQvip95XKk/gqcUr+m50vxgumxtA=
-----END CERTIFICATE-----`
	if _, err := file.Write([]byte(caCert)); err != nil {
		t.Fatalf("could not write to tempfile my.ca: %v", err)
	}

	encodedCAData := base64.StdEncoding.EncodeToString([]byte(caCert))

	testData := []byte(`
apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority: ` + file.Name() + `
    server: https://cluster-a.com
  name: cluster-a
- cluster:
    certificate-authority-data: ` + encodedCAData + `
    server: https://cluster-b.com
  name: cluster-b
contexts:
- context:
    cluster: cluster-a
    namespace: ns-a
    user: user-a
  name: context-a
- context:
    cluster: cluster-b
    namespace: ns-b
    user: user-b
  name: context-b
current-context: context-b
users:
- name: user-a
  user:
    token: mytoken-a
- name: user-b
  user:
    token: mytoken-b
`)
	f, err := os.CreateTemp("", "kubeconfig")
	if err != nil {
		t.Fatal(err)
	}
	// defer utiltesting.CloseAndRemove(t, f)
	f.Write(testData)

	config, err := loadRESTClientConfig(f.Name())
	if err != nil {
		t.Fatal(err)
	}

	expectedConfig := &restclient.Config{
		Host: "https://cluster-b.com",
		TLSClientConfig: restclient.TLSClientConfig{
			CAData: []byte(caCert),
		},
		BearerToken: "mytoken-b",
	}

	if !reflect.DeepEqual(config, expectedConfig) {
		t.Errorf("Unexpected config: %s", cmp.Diff(config, expectedConfig))
	}
}

func TestRequestNodeCertificateNoKeyData(t *testing.T) {
	certData, err := requestNodeCertificate(context.TODO(), newClientset(fakeClient{}), []byte{}, "fake-node-name")
	if err == nil {
		t.Errorf("Got no error, wanted error an error because there was an empty private key passed in.")
	}
	if certData != nil {
		t.Errorf("Got cert data, wanted nothing as there should have been an error.")
	}
}

func TestRequestNodeCertificateErrorCreatingCSR(t *testing.T) {
	client := newClientset(fakeClient{
		failureType: createError,
	})
	privateKeyData, err := keyutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		t.Fatalf("Unable to generate a new private key: %v", err)
	}

	certData, err := requestNodeCertificate(context.TODO(), client, privateKeyData, "fake-node-name")
	if err == nil {
		t.Errorf("Got no error, wanted error an error because client.Create failed.")
	}
	if certData != nil {
		t.Errorf("Got cert data, wanted nothing as there should have been an error.")
	}
}

func TestRequestNodeCertificate(t *testing.T) {
	privateKeyData, err := keyutil.MakeEllipticPrivateKeyPEM()
	if err != nil {
		t.Fatalf("Unable to generate a new private key: %v", err)
	}

	certData, err := requestNodeCertificate(context.TODO(), newClientset(fakeClient{}), privateKeyData, "fake-node-name")
	if err != nil {
		t.Errorf("Got %v, wanted no error.", err)
	}
	if certData == nil {
		t.Errorf("Got nothing, expected a CSR.")
	}
}

type failureType int

const (
	noError failureType = iota //nolint:deadcode,varcheck
	createError
	certificateSigningRequestDenied
)

type fakeClient struct {
	certificatesclient.CertificateSigningRequestInterface
	failureType failureType
}

func newClientset(opts fakeClient) *fake.Clientset {
	f := fake.NewSimpleClientset()
	switch opts.failureType {
	case createError:
		f.PrependReactor("create", "certificatesigningrequests", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
			switch action.GetResource().Version {
			case "v1":
				return true, nil, fmt.Errorf("create error")
			default:
				return true, nil, apierrors.NewNotFound(certificatesv1.Resource("certificatesigningrequests"), "")
			}
		})
	default:
		f.PrependReactor("create", "certificatesigningrequests", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
			switch action.GetResource().Version {
			case "v1":
				return true, &certificatesv1.CertificateSigningRequest{ObjectMeta: metav1.ObjectMeta{Name: "fake-certificate-signing-request-name", UID: "fake-uid"}}, nil
			default:
				return true, nil, apierrors.NewNotFound(certificatesv1.Resource("certificatesigningrequests"), "")
			}
		})
		f.PrependReactor("list", "certificatesigningrequests", func(action clienttesting.Action) (handled bool, ret runtime.Object, err error) {
			switch action.GetResource().Version {
			case "v1":
				return true, &certificatesv1.CertificateSigningRequestList{Items: []certificatesv1.CertificateSigningRequest{{ObjectMeta: metav1.ObjectMeta{Name: "fake-certificate-signing-request-name", UID: "fake-uid"}}}}, nil
			default:
				return true, nil, apierrors.NewNotFound(certificatesv1.Resource("certificatesigningrequests"), "")
			}
		})
		f.PrependWatchReactor("certificatesigningrequests", func(action clienttesting.Action) (handled bool, ret watch.Interface, err error) {
			switch action.GetResource().Version {
			case "v1":
				w := watch.NewFakeWithChanSize(1, false)
				w.Add(opts.generateCSR())
				w.Stop()
				return true, w, nil

			default:
				return true, nil, apierrors.NewNotFound(certificatesv1.Resource("certificatesigningrequests"), "")
			}
		})
	}
	return f
}

func (c fakeClient) generateCSR() runtime.Object {
	var condition certificatesv1.CertificateSigningRequestCondition
	var certificateData []byte
	if c.failureType == certificateSigningRequestDenied {
		condition = certificatesv1.CertificateSigningRequestCondition{
			Type: certificatesv1.CertificateDenied,
		}
	} else {
		condition = certificatesv1.CertificateSigningRequestCondition{
			Type: certificatesv1.CertificateApproved,
		}
		certificateData = []byte(`issued certificate`)
	}

	csr := certificatesv1.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			UID: "fake-uid",
		},
		Status: certificatesv1.CertificateSigningRequestStatus{
			Conditions: []certificatesv1.CertificateSigningRequestCondition{
				condition,
			},
			Certificate: certificateData,
		},
	}
	return &csr
}
