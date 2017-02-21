/*
Copyright 2017 The Kubernetes Authors.

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

package master

import (
	"encoding/json"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/kubernetes/pkg/api"
	coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
)

type ClientCARegistrationHook struct {
	ClientCA               []byte
	FrontProxyCA           []byte
	FrontProxyAllowedNames []string
}

func (h ClientCARegistrationHook) PostStartHook(hookContext genericapiserver.PostStartHookContext) error {
	if len(h.ClientCA) == 0 && len(h.FrontProxyCA) == 0 {
		return nil
	}

	client, err := coreclient.NewForConfig(hookContext.LoopbackClientConfig)
	if err != nil {
		utilruntime.HandleError(err)
		return nil
	}

	h.writeClientCAs(client)
	return nil
}

// writeClientCAs is here for unit testing with a fake client
func (h ClientCARegistrationHook) writeClientCAs(client coreclient.CoreInterface) {
	if _, err := client.Namespaces().Create(&api.Namespace{ObjectMeta: metav1.ObjectMeta{Name: metav1.NamespacePublic}}); err != nil && !apierrors.IsAlreadyExists(err) {
		utilruntime.HandleError(err)
		return
	}

	if len(h.ClientCA) > 0 {
		if err := writePublicClientCert(client, "client-ca", map[string]string{
			"client-ca.crt": string(h.ClientCA),
		}); err != nil {
			utilruntime.HandleError(err)
		}
	}

	if len(h.FrontProxyCA) > 0 {
		serializedNames, err := json.Marshal(h.FrontProxyAllowedNames)
		if err != nil {
			utilruntime.HandleError(err)
			return
		}
		data := map[string]string{
			"front-proxy-ca.crt":        string(h.FrontProxyCA),
			"front-proxy-allowed-names": string(serializedNames),
		}

		if err := writePublicClientCert(client, "front-proxy-ca", data); err != nil {
			utilruntime.HandleError(err)
		}
	}

	return
}

func writePublicClientCert(client coreclient.ConfigMapsGetter, name string, data map[string]string) error {
	existing, err := client.ConfigMaps(metav1.NamespacePublic).Get(name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		_, err := client.ConfigMaps(metav1.NamespacePublic).Create(&api.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespacePublic, Name: name},
			Data:       data,
		})
		return err
	}
	if err != nil {
		return err
	}

	existing.Data = data
	_, err = client.ConfigMaps(metav1.NamespacePublic).Update(existing)
	return err
}
