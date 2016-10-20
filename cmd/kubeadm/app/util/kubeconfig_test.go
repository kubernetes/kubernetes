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

package util

import (
        "fmt"
        "log"
        "math/rand"
        "os"
        "runtime"
        "strings"
        "testing"
        "time"

        clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
)

type configClient struct {
        c  string
        s  string
        ca []byte
}

type configClientWithCerts struct {
        c           *clientcmdapi.Config
        clusterName string
        userName    string
        clientKey   []byte
        clientCert  []byte
}

type configClientWithToken struct {
        c           *clientcmdapi.Config
        clusterName string
        userName    string
        token       string
}

func init() {
        rand.Seed(time.Now().UTC().UnixNano())
}

func TestCreateBasicClientConfig(t *testing.T) {
        var createBasicTest = []struct {
                cc       configClient
                expected string
        }{
                {configClient{}, ""},
                {configClient{c: "kubernetes"}, ""},
        }
        for _, rt := range createBasicTest {
                c := CreateBasicClientConfig(rt.cc.c, rt.cc.s, rt.cc.ca)
                if c.Kind != rt.expected {
                        t.Errorf(
                                "failed CreateBasicClientConfig:\n\texpected: %s\n\t  actual: %s",
                                c.Kind,
                                rt.expected,
                        )
                }
        }
}

func TestMakeClientConfigWithCerts(t *testing.T) {
        var createBasicTest = []struct {
                cc          configClient
                ccWithCerts configClientWithCerts
                expected    string
        }{
                {configClient{}, configClientWithCerts{}, ""},
                {configClient{c: "kubernetes"}, configClientWithCerts{}, ""},
        }
        for _, rt := range createBasicTest {
                c := CreateBasicClientConfig(rt.cc.c, rt.cc.s, rt.cc.ca)
                rt.ccWithCerts.c = c
                cwc := MakeClientConfigWithCerts(
                        rt.ccWithCerts.c,
                        rt.ccWithCerts.clusterName,
                        rt.ccWithCerts.userName,
                        rt.ccWithCerts.clientKey,
                        rt.ccWithCerts.clientCert,
                )
                if cwc.Kind != rt.expected {
                        t.Errorf(
                                "failed MakeClientConfigWithCerts:\n\texpected: %s\n\t  actual: %s",
                                c.Kind,
                                rt.expected,
                        )
                }
        }
}

func TestMakeClientConfigWithToken(t *testing.T) {
        var createBasicTest = []struct {
                cc          configClient
                ccWithToken configClientWithToken
                expected    string
        }{
                {configClient{}, configClientWithToken{}, ""},
                {configClient{c: "kubernetes"}, configClientWithToken{}, ""},
        }
        for _, rt := range createBasicTest {
                c := CreateBasicClientConfig(rt.cc.c, rt.cc.s, rt.cc.ca)
                rt.ccWithToken.c = c
                cwc := MakeClientConfigWithToken(
                        rt.ccWithToken.c,
                        rt.ccWithToken.clusterName,
                        rt.ccWithToken.userName,
                        rt.ccWithToken.token,
                )
                if cwc.Kind != rt.expected {
                        t.Errorf(
                                "failed MakeClientConfigWithCerts:\n\texpected: %s\n\t  actual: %s",
                                c.Kind,
                                rt.expected,
                        )
                }
        }
}

func setEnvs() {
        rand := rand.Int()
        var envParams = map[string]string{
                "kubernetes_dir":     fmt.Sprintf("/tmp/%d/etc/kubernetes", rand),
                "host_pki_path":      fmt.Sprintf("/tmp/%d/etc/kubernetes/pki", rand),
                "host_etcd_path":     fmt.Sprintf("/tmp/%d/var/lib/etcd", rand),
                "hyperkube_image":    "",
                "discovery_image":    fmt.Sprintf("gcr.io/google_containers/kube-discovery-%s:%s", runtime.GOARCH, "1.0"),
                "etcd_image":         "",
                "component_loglevel": "--v=4",
        }
        for k, v := range envParams {
                err := os.Setenv(fmt.Sprintf("KUBE_%s", strings.ToUpper(k)), v)
                if err != nil {
                        log.Fatal(err)
                }
        }
}

func TestWriteKubeconfigIfNotExists(t *testing.T) {
        setEnvs()
        var writeConfig = []struct {
                name     string
                cc       configClient
                expected error
        }{
                {fmt.Sprintf("%d", rand.Int()), configClient{}, nil},
                {fmt.Sprintf("%d", rand.Int()), configClient{c: "kubernetes"}, nil},
        }
        for _, rt := range writeConfig {
                c := CreateBasicClientConfig(rt.cc.c, rt.cc.s, rt.cc.ca)
                err := WriteKubeconfigIfNotExists(rt.name, c)
                if err != rt.expected {
                        t.Errorf(
                                "failed WriteKubeconfigIfNotExists:\n\texpected: %s\n\t  actual: %s",
                                err,
                                rt.expected,
                        )
                }
        }
}
