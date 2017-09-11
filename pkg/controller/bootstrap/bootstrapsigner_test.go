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
	"testing"

	"github.com/davecgh/go-spew/spew"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
)

func init() {
	spew.Config.DisableMethods = true
}

const testTokenID = "abc123"

func newBootstrapSigner() (*BootstrapSigner, *fake.Clientset) {
	options := DefaultBootstrapSignerOptions()
	cl := fake.NewSimpleClientset()
	return NewBootstrapSigner(cl, options), cl
}

func newConfigMap(tokenID, signature string) *v1.ConfigMap {
	ret := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:       metav1.NamespacePublic,
			Name:            bootstrapapi.ConfigMapClusterInfo,
			ResourceVersion: "1",
		},
		Data: map[string]string{
			bootstrapapi.KubeConfigKey: "payload",
		},
	}
	if len(tokenID) > 0 {
		ret.Data[bootstrapapi.JWSSignatureKeyPrefix+tokenID] = signature
	}
	return ret
}

func TestNoConfigMap(t *testing.T) {
	signer, cl := newBootstrapSigner()
	signer.signConfigMap()
	verifyActions(t, []core.Action{}, cl.Actions())
}

func TestSimpleSign(t *testing.T) {
	signer, cl := newBootstrapSigner()

	cm := newConfigMap("", "")
	signer.configMaps.Add(cm)

	secret := newTokenSecret(testTokenID, "tokenSecret")
	addSecretSigningUsage(secret, "true")
	signer.secrets.Add(secret)

	signer.signConfigMap()

	expected := []core.Action{
		core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "configmaps"},
			api.NamespacePublic,
			newConfigMap(testTokenID, "eyJhbGciOiJIUzI1NiIsImtpZCI6ImFiYzEyMyJ9..QSxpUG7Q542CirTI2ECPSZjvBOJURUW5a7XqFpNI958")),
	}

	verifyActions(t, expected, cl.Actions())
}

func TestNoSignNeeded(t *testing.T) {
	signer, cl := newBootstrapSigner()

	cm := newConfigMap(testTokenID, "eyJhbGciOiJIUzI1NiIsImtpZCI6ImFiYzEyMyJ9..QSxpUG7Q542CirTI2ECPSZjvBOJURUW5a7XqFpNI958")
	signer.configMaps.Add(cm)

	secret := newTokenSecret(testTokenID, "tokenSecret")
	addSecretSigningUsage(secret, "true")
	signer.secrets.Add(secret)

	signer.signConfigMap()

	verifyActions(t, []core.Action{}, cl.Actions())
}

func TestUpdateSignature(t *testing.T) {
	signer, cl := newBootstrapSigner()

	cm := newConfigMap(testTokenID, "old signature")
	signer.configMaps.Add(cm)

	secret := newTokenSecret(testTokenID, "tokenSecret")
	addSecretSigningUsage(secret, "true")
	signer.secrets.Add(secret)

	signer.signConfigMap()

	expected := []core.Action{
		core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "configmaps"},
			api.NamespacePublic,
			newConfigMap(testTokenID, "eyJhbGciOiJIUzI1NiIsImtpZCI6ImFiYzEyMyJ9..QSxpUG7Q542CirTI2ECPSZjvBOJURUW5a7XqFpNI958")),
	}

	verifyActions(t, expected, cl.Actions())
}

func TestRemoveSignature(t *testing.T) {
	signer, cl := newBootstrapSigner()

	cm := newConfigMap(testTokenID, "old signature")
	signer.configMaps.Add(cm)

	signer.signConfigMap()

	expected := []core.Action{
		core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "configmaps"},
			api.NamespacePublic,
			newConfigMap("", "")),
	}

	verifyActions(t, expected, cl.Actions())
}
