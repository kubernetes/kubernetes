/*
Copyright 2015 The Kubernetes Authors.

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

package api

import (
	"bytes"
	"fmt"
	"os"
	"reflect"
	"testing"

	utiltesting "k8s.io/client-go/util/testing"

	"sigs.k8s.io/yaml"
)

func newMergedConfig(certFile, certContent, keyFile, keyContent, caFile, caContent string, t *testing.T) Config {
	if err := os.WriteFile(certFile, []byte(certContent), 0644); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := os.WriteFile(keyFile, []byte(keyContent), 0600); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if err := os.WriteFile(caFile, []byte(caContent), 0644); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	return Config{
		AuthInfos: map[string]*AuthInfo{
			"red-user":  {Token: "red-token", ClientCertificateData: []byte(certContent), ClientKeyData: []byte(keyContent)},
			"blue-user": {Token: "blue-token", ClientCertificate: certFile, ClientKey: keyFile}},
		Clusters: map[string]*Cluster{
			"cow-cluster":     {Server: "http://cow.org:8080", CertificateAuthorityData: []byte(caContent)},
			"chicken-cluster": {Server: "http://chicken.org:8080", CertificateAuthority: caFile}},
		Contexts: map[string]*Context{
			"federal-context": {AuthInfo: "red-user", Cluster: "cow-cluster"},
			"shaker-context":  {AuthInfo: "blue-user", Cluster: "chicken-cluster"}},
		CurrentContext: "federal-context",
	}
}

func TestMinifySuccess(t *testing.T) {
	certFile, _ := os.CreateTemp("", "")
	keyFile, _ := os.CreateTemp("", "")
	caFile, _ := os.CreateTemp("", "")
	defer utiltesting.CloseAndRemove(t, certFile, keyFile, caFile)

	mutatingConfig := newMergedConfig(certFile.Name(), "cert", keyFile.Name(), "key", caFile.Name(), "ca", t)

	if err := MinifyConfig(&mutatingConfig); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(mutatingConfig.Contexts) > 1 {
		t.Errorf("unexpected contexts: %v", mutatingConfig.Contexts)
	}
	if _, exists := mutatingConfig.Contexts["federal-context"]; !exists {
		t.Errorf("missing context")
	}

	if len(mutatingConfig.Clusters) > 1 {
		t.Errorf("unexpected clusters: %v", mutatingConfig.Clusters)
	}
	if _, exists := mutatingConfig.Clusters["cow-cluster"]; !exists {
		t.Errorf("missing cluster")
	}

	if len(mutatingConfig.AuthInfos) > 1 {
		t.Errorf("unexpected users: %v", mutatingConfig.AuthInfos)
	}
	if _, exists := mutatingConfig.AuthInfos["red-user"]; !exists {
		t.Errorf("missing user")
	}
}

func TestMinifyMissingContext(t *testing.T) {
	certFile, _ := os.CreateTemp("", "")
	keyFile, _ := os.CreateTemp("", "")
	caFile, _ := os.CreateTemp("", "")
	defer utiltesting.CloseAndRemove(t, certFile, keyFile, caFile)

	mutatingConfig := newMergedConfig(certFile.Name(), "cert", keyFile.Name(), "key", caFile.Name(), "ca", t)
	mutatingConfig.CurrentContext = "missing"

	errMsg := "cannot locate context missing"

	if err := MinifyConfig(&mutatingConfig); err == nil || err.Error() != errMsg {
		t.Errorf("expected %v, got %v", errMsg, err)
	}
}

func TestMinifyMissingCluster(t *testing.T) {
	certFile, _ := os.CreateTemp("", "")
	keyFile, _ := os.CreateTemp("", "")
	caFile, _ := os.CreateTemp("", "")
	defer utiltesting.CloseAndRemove(t, certFile, keyFile, caFile)

	mutatingConfig := newMergedConfig(certFile.Name(), "cert", keyFile.Name(), "key", caFile.Name(), "ca", t)
	delete(mutatingConfig.Clusters, mutatingConfig.Contexts[mutatingConfig.CurrentContext].Cluster)

	errMsg := "cannot locate cluster cow-cluster"

	if err := MinifyConfig(&mutatingConfig); err == nil || err.Error() != errMsg {
		t.Errorf("expected %v, got %v", errMsg, err)
	}
}

func TestMinifyMissingAuthInfo(t *testing.T) {
	certFile, _ := os.CreateTemp("", "")
	keyFile, _ := os.CreateTemp("", "")
	caFile, _ := os.CreateTemp("", "")
	defer utiltesting.CloseAndRemove(t, certFile, keyFile, caFile)

	mutatingConfig := newMergedConfig(certFile.Name(), "cert", keyFile.Name(), "key", caFile.Name(), "ca", t)
	delete(mutatingConfig.AuthInfos, mutatingConfig.Contexts[mutatingConfig.CurrentContext].AuthInfo)

	errMsg := "cannot locate user red-user"

	if err := MinifyConfig(&mutatingConfig); err == nil || err.Error() != errMsg {
		t.Errorf("expected %v, got %v", errMsg, err)
	}
}

func TestFlattenSuccess(t *testing.T) {
	certFile, _ := os.CreateTemp("", "")
	keyFile, _ := os.CreateTemp("", "")
	caFile, _ := os.CreateTemp("", "")
	defer utiltesting.CloseAndRemove(t, certFile, keyFile, caFile)

	certData := "cert"
	keyData := "key"
	caData := "ca"

	unchangingCluster := "cow-cluster"
	unchangingAuthInfo := "red-user"
	changingCluster := "chicken-cluster"
	changingAuthInfo := "blue-user"

	startingConfig := newMergedConfig(certFile.Name(), certData, keyFile.Name(), keyData, caFile.Name(), caData, t)
	mutatingConfig := startingConfig

	if err := FlattenConfig(&mutatingConfig); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(mutatingConfig.Contexts) != 2 {
		t.Errorf("unexpected contexts: %v", mutatingConfig.Contexts)
	}
	if !reflect.DeepEqual(startingConfig.Contexts, mutatingConfig.Contexts) {
		t.Errorf("expected %v, got %v", startingConfig.Contexts, mutatingConfig.Contexts)
	}

	if len(mutatingConfig.Clusters) != 2 {
		t.Errorf("unexpected clusters: %v", mutatingConfig.Clusters)
	}
	if !reflect.DeepEqual(startingConfig.Clusters[unchangingCluster], mutatingConfig.Clusters[unchangingCluster]) {
		t.Errorf("expected %v, got %v", startingConfig.Clusters[unchangingCluster], mutatingConfig.Clusters[unchangingCluster])
	}
	if len(mutatingConfig.Clusters[changingCluster].CertificateAuthority) != 0 {
		t.Errorf("unexpected caFile")
	}
	if string(mutatingConfig.Clusters[changingCluster].CertificateAuthorityData) != caData {
		t.Errorf("expected %v, got %v", caData, string(mutatingConfig.Clusters[changingCluster].CertificateAuthorityData))
	}

	if len(mutatingConfig.AuthInfos) != 2 {
		t.Errorf("unexpected users: %v", mutatingConfig.AuthInfos)
	}
	if !reflect.DeepEqual(startingConfig.AuthInfos[unchangingAuthInfo], mutatingConfig.AuthInfos[unchangingAuthInfo]) {
		t.Errorf("expected %v, got %v", startingConfig.AuthInfos[unchangingAuthInfo], mutatingConfig.AuthInfos[unchangingAuthInfo])
	}
	if len(mutatingConfig.AuthInfos[changingAuthInfo].ClientCertificate) != 0 {
		t.Errorf("unexpected caFile")
	}
	if string(mutatingConfig.AuthInfos[changingAuthInfo].ClientCertificateData) != certData {
		t.Errorf("expected %v, got %v", certData, string(mutatingConfig.AuthInfos[changingAuthInfo].ClientCertificateData))
	}
	if len(mutatingConfig.AuthInfos[changingAuthInfo].ClientKey) != 0 {
		t.Errorf("unexpected caFile")
	}
	if string(mutatingConfig.AuthInfos[changingAuthInfo].ClientKeyData) != keyData {
		t.Errorf("expected %v, got %v", keyData, string(mutatingConfig.AuthInfos[changingAuthInfo].ClientKeyData))
	}

}

func Example_minifyAndShorten() {
	certFile, _ := os.CreateTemp("", "")
	keyFile, _ := os.CreateTemp("", "")
	caFile, _ := os.CreateTemp("", "")
	defer utiltesting.CloseAndRemove(&testing.T{}, certFile, keyFile, caFile)

	certData := "cert"
	keyData := "key"
	caData := "ca"

	config := newMergedConfig(certFile.Name(), certData, keyFile.Name(), keyData, caFile.Name(), caData, nil)

	MinifyConfig(&config)
	ShortenConfig(&config)

	output, _ := yaml.Marshal(config)
	fmt.Printf("%s", string(output))
	// Output:
	// clusters:
	//   cow-cluster:
	//     certificate-authority-data: DATA+OMITTED
	//     server: http://cow.org:8080
	// contexts:
	//   federal-context:
	//     cluster: cow-cluster
	//     user: red-user
	// current-context: federal-context
	// preferences: {}
	// users:
	//   red-user:
	//     client-certificate-data: DATA+OMITTED
	//     client-key-data: DATA+OMITTED
	//     token: REDACTED
}

func TestShortenSuccess(t *testing.T) {
	certFile, _ := os.CreateTemp("", "")
	keyFile, _ := os.CreateTemp("", "")
	caFile, _ := os.CreateTemp("", "")
	defer utiltesting.CloseAndRemove(t, certFile, keyFile, caFile)

	certData := "cert"
	keyData := "key"
	caData := "ca"

	unchangingCluster := "chicken-cluster"
	unchangingAuthInfo := "blue-user"
	changingCluster := "cow-cluster"
	changingAuthInfo := "red-user"

	startingConfig := newMergedConfig(certFile.Name(), certData, keyFile.Name(), keyData, caFile.Name(), caData, t)
	mutatingConfig := startingConfig

	ShortenConfig(&mutatingConfig)

	if len(mutatingConfig.Contexts) != 2 {
		t.Errorf("unexpected contexts: %v", mutatingConfig.Contexts)
	}
	if !reflect.DeepEqual(startingConfig.Contexts, mutatingConfig.Contexts) {
		t.Errorf("expected %v, got %v", startingConfig.Contexts, mutatingConfig.Contexts)
	}

	dataOmitted := string(dataOmittedBytes)
	if len(mutatingConfig.Clusters) != 2 {
		t.Errorf("unexpected clusters: %v", mutatingConfig.Clusters)
	}
	if !reflect.DeepEqual(startingConfig.Clusters[unchangingCluster], mutatingConfig.Clusters[unchangingCluster]) {
		t.Errorf("expected %v, got %v", startingConfig.Clusters[unchangingCluster], mutatingConfig.Clusters[unchangingCluster])
	}
	if string(mutatingConfig.Clusters[changingCluster].CertificateAuthorityData) != dataOmitted {
		t.Errorf("expected %v, got %v", dataOmitted, string(mutatingConfig.Clusters[changingCluster].CertificateAuthorityData))
	}

	if len(mutatingConfig.AuthInfos) != 2 {
		t.Errorf("unexpected users: %v", mutatingConfig.AuthInfos)
	}
	if !reflect.DeepEqual(startingConfig.AuthInfos[unchangingAuthInfo], mutatingConfig.AuthInfos[unchangingAuthInfo]) {
		t.Errorf("expected %v, got %v", startingConfig.AuthInfos[unchangingAuthInfo], mutatingConfig.AuthInfos[unchangingAuthInfo])
	}
	if string(mutatingConfig.AuthInfos[changingAuthInfo].ClientCertificateData) != dataOmitted {
		t.Errorf("expected %v, got %v", dataOmitted, string(mutatingConfig.AuthInfos[changingAuthInfo].ClientCertificateData))
	}
	if string(mutatingConfig.AuthInfos[changingAuthInfo].ClientKeyData) != dataOmitted {
		t.Errorf("expected %v, got %v", dataOmitted, string(mutatingConfig.AuthInfos[changingAuthInfo].ClientKeyData))
	}
	if mutatingConfig.AuthInfos[changingAuthInfo].Token != "REDACTED" {
		t.Errorf("expected REDACTED, got %q", mutatingConfig.AuthInfos[changingAuthInfo].Token)
	}
}

func TestRedactSecrets(t *testing.T) {
	certFile, _ := os.CreateTemp("", "")
	defer os.Remove(certFile.Name())
	keyFile, _ := os.CreateTemp("", "")
	defer os.Remove(keyFile.Name())
	caFile, _ := os.CreateTemp("", "")
	defer os.Remove(caFile.Name())

	certData := "cert"
	keyData := "key"
	caData := "ca"

	unchangingCluster := "chicken-cluster"
	unchangingAuthInfo := "blue-user"
	changingAuthInfo := "red-user"

	startingConfig := newMergedConfig(certFile.Name(), certData, keyFile.Name(), keyData, caFile.Name(), caData, t)
	mutatingConfig := startingConfig

	err := RedactSecrets(&mutatingConfig)
	if err != nil {
		t.Errorf("unexpected error redacting secrets:\n%v", err)
	}

	if len(mutatingConfig.Contexts) != 2 {
		t.Errorf("unexpected contexts: %v", mutatingConfig.Contexts)
	}
	if !reflect.DeepEqual(startingConfig.Contexts, mutatingConfig.Contexts) {
		t.Errorf("expected %v, got %v", startingConfig.Contexts, mutatingConfig.Contexts)
	}

	if len(mutatingConfig.Clusters) != 2 {
		t.Errorf("unexpected clusters: %v", mutatingConfig.Clusters)
	}
	if !reflect.DeepEqual(startingConfig.Clusters[unchangingCluster], mutatingConfig.Clusters[unchangingCluster]) {
		t.Errorf("expected %v, got %v", startingConfig.Clusters[unchangingCluster], mutatingConfig.Clusters[unchangingCluster])
	}

	if len(mutatingConfig.AuthInfos) != 2 {
		t.Errorf("unexpected users: %v", mutatingConfig.AuthInfos)
	}
	if !reflect.DeepEqual(startingConfig.AuthInfos[unchangingAuthInfo], mutatingConfig.AuthInfos[unchangingAuthInfo]) {
		t.Errorf("expected %v, got %v", startingConfig.AuthInfos[unchangingAuthInfo], mutatingConfig.AuthInfos[unchangingAuthInfo])
	}
	if mutatingConfig.AuthInfos[changingAuthInfo].Token != "REDACTED" {
		t.Errorf("expected REDACTED, got %v", mutatingConfig.AuthInfos[changingAuthInfo].Token)
	}
	if !bytes.Equal(mutatingConfig.AuthInfos[changingAuthInfo].ClientKeyData, []byte("REDACTED")) {
		t.Errorf("expected REDACTED, got %s", mutatingConfig.AuthInfos[changingAuthInfo].ClientKeyData)
	}
}
