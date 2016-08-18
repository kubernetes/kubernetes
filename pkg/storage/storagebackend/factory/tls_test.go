package factory

import (
	"io/ioutil"
	"os"
	"path"
	"testing"

	"golang.org/x/net/context"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/storage/etcd/testing/testingcert"
	"k8s.io/kubernetes/pkg/storage/storagebackend"

	"github.com/coreos/etcd/integration"
	"github.com/coreos/etcd/pkg/transport"
)

func TestTLSConnection(t *testing.T) {
	certFile, keyFile, caFile := configureTLSCerts(t)

	tlsInfo := &transport.TLSInfo{
		CertFile: certFile,
		KeyFile:  keyFile,
		CAFile:   caFile,
	}

	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{
		Size:      1,
		ClientTLS: tlsInfo,
	})
	defer cluster.Terminate(t)

	cfg := storagebackend.Config{
		Type:       storagebackend.StorageTypeETCD3,
		ServerList: []string{cluster.Members[0].GRPCAddr()},
		CertFile:   certFile,
		KeyFile:    keyFile,
		CAFile:     caFile,
		Codec:      testapi.Default.Codec(),
	}
	storage, err := newETCD3Storage(cfg)
	if err != nil {
		t.Fatal(err)
	}
	err = storage.Create(context.TODO(), "/abc", &api.Pod{}, nil, 0)
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
