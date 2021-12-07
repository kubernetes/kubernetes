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

package factory

import (
	"context"
	"io/ioutil"
	"net"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"testing"
	"time"

	"go.etcd.io/etcd/client/pkg/v3/transport"

	apitesting "k8s.io/apimachinery/pkg/api/apitesting"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage/etcd3/testing/testingcert"
	"k8s.io/apiserver/pkg/storage/etcd3/testserver"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

var scheme = runtime.NewScheme()
var codecs = serializer.NewCodecFactory(scheme)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
}

func TestTLSConnection(t *testing.T) {
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)

	certFile, keyFile, caFile := configureTLSCerts(t)
	defer os.RemoveAll(filepath.Dir(certFile))

	// override server config to be TLS-enabled
	etcdConfig := testserver.NewTestConfig(t)
	etcdConfig.ClientTLSInfo = transport.TLSInfo{
		CertFile:      certFile,
		KeyFile:       keyFile,
		TrustedCAFile: caFile,
	}
	for i := range etcdConfig.LCUrls {
		etcdConfig.LCUrls[i].Scheme = "https"
	}
	for i := range etcdConfig.ACUrls {
		etcdConfig.ACUrls[i].Scheme = "https"
	}

	client := testserver.RunEtcd(t, etcdConfig)
	cfg := storagebackend.Config{
		Type: storagebackend.StorageTypeETCD3,
		Transport: storagebackend.TransportConfig{
			ServerList:    client.Endpoints(),
			CertFile:      certFile,
			KeyFile:       keyFile,
			TrustedCAFile: caFile,
		},
		Codec: codec,
	}
	storage, destroyFunc, err := newETCD3Storage(*cfg.ForResource(schema.GroupResource{Resource: "pods"}), nil)
	defer destroyFunc()
	if err != nil {
		t.Fatal(err)
	}
	err = storage.Create(context.TODO(), "/abc", &example.Pod{}, nil, 0)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
}

func TestTLSConnectionSetServerName(t *testing.T) {
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)

	certFile, keyFile, caFile := configureTLSCerts(t)
	defer os.RemoveAll(filepath.Dir(certFile))

	// override server config to be TLS-enabled
	etcdConfig := testserver.NewTestConfig(t)
	etcdConfig.ClientTLSInfo = transport.TLSInfo{
		CertFile:      certFile,
		KeyFile:       keyFile,
		TrustedCAFile: caFile,
	}

	lcUrl := etcdConfig.LCUrls[0]
	lcUrl.Scheme = "https"
	serverList := []string{lcUrl.String()}
	// override listen-client-urls to be [localhost, 127.0.1.1],
	// so that the etcd client could both make the initial connection
	// through serverList (localhost), and later on after AutoSync
	// has run (127.0.1.1).
	lcUrls := []url.URL{lcUrl}
	lcUrl.Host = replaceHost(t, lcUrl.Host, "127.0.1.1")
	lcUrls = append(lcUrls, lcUrl)
	etcdConfig.LCUrls = lcUrls

	acUrl := etcdConfig.ACUrls[0]
	acUrl.Scheme = "https"
	// override advertise-client-urls to be an address
	// not valid for the certificate
	acUrl.Host = replaceHost(t, acUrl.Host, "127.0.1.1")
	etcdConfig.ACUrls = []url.URL{acUrl}

	// Set ServerName to be a name valid for the certificate so that
	// the client created in testserver.RunEtcd could connect to etcd
	serverUrl, err := url.Parse(serverList[0])
	if err != nil {
		t.Fatal(err)
	}
	serverHost, _, err := net.SplitHostPort(serverUrl.Host)
	if err != nil {
		t.Fatal(err)
	}
	etcdConfig.ClientTLSInfo.ServerName = serverHost

	_ = testserver.RunEtcd(t, etcdConfig)

	cfg := storagebackend.Config{
		Type: storagebackend.StorageTypeETCD3,
		Transport: storagebackend.TransportConfig{
			ServerList:       serverList,
			CertFile:         certFile,
			KeyFile:          keyFile,
			TrustedCAFile:    caFile,
			AutoSyncInterval: time.Second,
			SetServerName:    true,
		},
		Codec: codec,
	}
	storage, destroyFunc, err := newETCD3Storage(*cfg.ForResource(schema.GroupResource{Resource: "pods"}), nil)

	defer destroyFunc()
	if err != nil {
		t.Fatal(err)
	}

	// sleep 5 seconds so that AutoSync could trigger
	time.Sleep(5 * time.Second)
	ctx, cancelFunc := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelFunc()
	err = storage.Create(ctx, "/abc", &example.Pod{}, nil, 0)
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}
}

func configureTLSCerts(t *testing.T) (certFile, keyFile, caFile string) {
	baseDir := os.TempDir()
	tempDir, err := ioutil.TempDir(baseDir, "etcd_certificates")
	if err != nil {
		t.Fatal(err)
	}
	certFile = path.Join(tempDir, "etcdcert.pem")
	if err := ioutil.WriteFile(certFile, []byte(testingcert.CertFileContent), 0644); err != nil {
		t.Fatal(err)
	}
	keyFile = path.Join(tempDir, "etcdkey.pem")
	if err := ioutil.WriteFile(keyFile, []byte(testingcert.KeyFileContent), 0644); err != nil {
		t.Fatal(err)
	}
	caFile = path.Join(tempDir, "ca.pem")
	if err := ioutil.WriteFile(caFile, []byte(testingcert.CAFileContent), 0644); err != nil {
		t.Fatal(err)
	}
	return certFile, keyFile, caFile
}

func replaceHost(t *testing.T, hostPort, host string) string {
	_, port, err := net.SplitHostPort(hostPort)
	if err != nil {
		t.Fatal(err)
	}
	return net.JoinHostPort(host, port)
}
