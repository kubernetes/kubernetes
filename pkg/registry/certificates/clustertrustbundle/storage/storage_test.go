/*
Copyright 2022 The Kubernetes Authors.

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

package storage

import (
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

const validCert1 = `
-----BEGIN CERTIFICATE-----
MIIDmTCCAoGgAwIBAgIUUW9bIIsHU61w3yQR6amBuVvRFvcwDQYJKoZIhvcNAQEL
BQAwXDELMAkGA1UEBhMCeHgxCjAIBgNVBAgMAXgxCjAIBgNVBAcMAXgxCjAIBgNV
BAoMAXgxCjAIBgNVBAsMAXgxCzAJBgNVBAMMAmNhMRAwDgYJKoZIhvcNAQkBFgF4
MB4XDTIyMTAxODIzNTIyNFoXDTIzMTAxODIzNTIyNFowXDELMAkGA1UEBhMCeHgx
CjAIBgNVBAgMAXgxCjAIBgNVBAcMAXgxCjAIBgNVBAoMAXgxCjAIBgNVBAsMAXgx
CzAJBgNVBAMMAmNhMRAwDgYJKoZIhvcNAQkBFgF4MIIBIjANBgkqhkiG9w0BAQEF
AAOCAQ8AMIIBCgKCAQEA4PeK4SmlsNwpw97gTtjODQytUfyqhBIwdENwJUbc019Y
m3VTCRLCGXjUa22mV6/j7V+mZw114ePFYTiGAH+2dUzWAZOphvtzE5ttPuv6A6Zx
k2J69lNFwJ2fPd7XQIH7pEIXjiEBaszxKZKMsN9+jOGu6iFFAwYLMemFYDbZHuqb
OwdQcSEsy5wO2ANzFRuYzGXuNcS8jYLHftE8g2P+L0wXnV9eW6/lM2ZFxS/nzDJz
qtzrEvQrBsmskTNC8gCRRZ7askp3CVdPKjC90sxAPwhpi8JjJZxSe1Bn/WRHUz82
GFytEIJNx9hJY2GI316zkxgTbsxfRQe4QLJN7sRtpwIDAQABo1MwUTAdBgNVHQ4E
FgQU9FGsI8t+cu68fGkhtvO9FtUd174wHwYDVR0jBBgwFoAU9FGsI8t+cu68fGkh
tvO9FtUd174wDwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAqDIp
In5h2xZfEZcijT3mjfG8Bo6taxM2biy1M7wEpmDrElmrjMLsflZepcjgkSoVz9hP
cSX/k9ls1zy1H799gcjs+afSpIa1N0nUIxAKF1RHsFa+dvXpSA8YdhUnbEcBnqx0
vN2nDBFpdCSNf+EXNEj12+9ZJm6TLzx22f9vHyRCg4D36X3Rj1FCBWxhf0mSt3ek
5px3H53Xu42MqzZCiJc8/m+IqZHaixZS4bsayssaxif2fNxzAIZhgTygo8P8QGjI
rUmstMbg4PPq62x1yLAxEo+8XCg05saWZs384JE+K1SDqxobm51EROWVwi8jUrNC
9nojtkQ+jDZD+1Stiw==
-----END CERTIFICATE-----
`

const validCert2 = `
-----BEGIN CERTIFICATE-----
MIIC/jCCAeagAwIBAgIBADANBgkqhkiG9w0BAQsFADAVMRMwEQYDVQQDEwprdWJl
cm5ldGVzMB4XDTIyMTAxOTIzMTY0MFoXDTMyMTAxNjIzMTY0MFowFTETMBEGA1UE
AxMKa3ViZXJuZXRlczCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAO+k
zbj35jHIjCd5mxP1FHMwMtvLFPeKUjtaLDP9Bs2jZ97Igmr7NTysn9QZkRP68/XX
j993Y8tOLg71N4vRggWiYP+T9Xfo0uHZJmzADKx5XkuC4Gqv79dUdb8IKfAbX9HB
ffGmWRnZLLTu8Bv/vfyl0CfE64a57DK+CzNJDwdK46CYYUnEH6Wb9finYrMQ+PLG
Oi2c0J4KAYc1WTId5npNwouzf/IMD33PvuXfE7r+/pDbP8u/X03e7U0cc9l7KRxr
3gpRQemCG74yRuy1dd3lJ1YCD8q96xVVZimGebnJ0IHi+lORRa2ix/o3OzW3FaP+
6kzHU6VnBRDr2rAhMh0CAwEAAaNZMFcwDgYDVR0PAQH/BAQDAgKkMA8GA1UdEwEB
/wQFMAMBAf8wHQYDVR0OBBYEFGUVOLM74t1TVoZjifsLl3Rwt1A6MBUGA1UdEQQO
MAyCCmt1YmVybmV0ZXMwDQYJKoZIhvcNAQELBQADggEBANHnPVDemZqRybYPN1as
Ywxi3iT1I3Wma1rZyxTWeIq8Ik0gnyvbtCD1cFB/5QU1xPW09YnmIFM/E73RIeWT
RmCNMgOGmegYxBQRe4UvmwWGJzKNA66c0MBmd2LDHrQlrvdewOCR667Sm9krsGt1
tS/t6N/uBXeRSkXKEDXa+jOpYrV3Oq3IntG6zUeCrVbrH2Bs9Ma5fU00TwK3ylw5
Ww8KzYdQaxxrLaiRRtFcpM9dFH/vwxl1QUa5vjHcmUjxmZunEmXKplATyLT0FXDw
JAo8AuwuuwRh2o+o8SxwzzA+/EBrIREgcv5uIkD352QnfGkEvGu6JOPGZVyd/kVg
KA0=
-----END CERTIFICATE-----
`

func newStorage(t *testing.T) (*REST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorageForResource(t, certificates.SchemeGroupVersion.WithResource("clustertrustbundles").GroupResource())
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "clustertrustbundles",
	}
	storage, err := NewREST(restOptions)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return storage, server
}

func TestCreate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	validBundle := &certificates.ClusterTrustBundle{
		ObjectMeta: metav1.ObjectMeta{
			Name: "ctb1",
		},
		Spec: certificates.ClusterTrustBundleSpec{
			TrustBundle: validCert1,
		},
	}

	invalidBundle := &certificates.ClusterTrustBundle{
		ObjectMeta: metav1.ObjectMeta{
			Name: "ctb1",
		},
		Spec: certificates.ClusterTrustBundleSpec{
			// Empty TrustBundle is invalid.
		},
	}

	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()

	test.TestCreate(validBundle, invalidBundle)
}

func TestUpdate(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()

	test.TestUpdate(
		&certificates.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "ctb1",
			},
			Spec: certificates.ClusterTrustBundleSpec{
				TrustBundle: validCert1,
			},
		},
		// Valid update
		func(object runtime.Object) runtime.Object {
			bundle := object.(*certificates.ClusterTrustBundle)
			bundle.Spec.TrustBundle = strings.Join([]string{validCert1, validCert2}, "\n")
			return bundle
		},
		// Invalid update
		func(object runtime.Object) runtime.Object {
			bundle := object.(*certificates.ClusterTrustBundle)
			bundle.Spec.TrustBundle = ""
			return bundle
		},
	)
}

func TestDelete(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()

	test.TestDelete(
		&certificates.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "ctb1",
			},
			Spec: certificates.ClusterTrustBundleSpec{
				TrustBundle: validCert1,
			},
		},
	)
}

func TestGet(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()

	test.TestGet(
		&certificates.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "ctb1",
			},
			Spec: certificates.ClusterTrustBundleSpec{
				TrustBundle: validCert1,
			},
		},
	)
}

func TestList(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()

	test.TestList(
		&certificates.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "ctb1",
			},
			Spec: certificates.ClusterTrustBundleSpec{
				TrustBundle: validCert1,
			},
		},
	)
}

func TestWatch(t *testing.T) {
	storage, server := newStorage(t)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	test := genericregistrytest.New(t, storage.Store)
	test = test.ClusterScope()

	test.TestWatch(
		&certificates.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				Name: "ctb1",
			},
			Spec: certificates.ClusterTrustBundleSpec{
				SignerName:  "k8s.io/foo",
				TrustBundle: validCert1,
			},
		},
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "ctb1"},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}
