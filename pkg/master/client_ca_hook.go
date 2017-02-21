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
	ClientCA []byte

	RequestHeaderUsernameHeaders     []string
	RequestHeaderGroupHeaders        []string
	RequestHeaderExtraHeaderPrefixes []string
	RequestHeaderCA                  []byte
	RequestHeaderAllowedNames        []string
}

func (h ClientCARegistrationHook) PostStartHook(hookContext genericapiserver.PostStartHookContext) error {
	if len(h.ClientCA) == 0 && len(h.RequestHeaderCA) == 0 {
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
	if _, err := client.Namespaces().Create(&api.Namespace{ObjectMeta: metav1.ObjectMeta{Name: metav1.NamespaceSystem}}); err != nil && !apierrors.IsAlreadyExists(err) {
		utilruntime.HandleError(err)
		return
	}

	data := map[string]string{}
	if len(h.ClientCA) > 0 {
		data["client-ca-file"] = string(h.ClientCA)
	}

	if len(h.RequestHeaderCA) > 0 {
		var err error

		data["requestheader-username-headers"], err = jsonSerializeStringSlice(h.RequestHeaderUsernameHeaders)
		if err != nil {
			utilruntime.HandleError(err)
			return
		}
		data["requestheader-group-headers"], err = jsonSerializeStringSlice(h.RequestHeaderGroupHeaders)
		if err != nil {
			utilruntime.HandleError(err)
			return
		}
		data["requestheader-extra-headers-prefix"], err = jsonSerializeStringSlice(h.RequestHeaderExtraHeaderPrefixes)
		if err != nil {
			utilruntime.HandleError(err)
			return
		}
		data["requestheader-client-ca-file"] = string(h.RequestHeaderCA)
		data["requestheader-allowed-names"], err = jsonSerializeStringSlice(h.RequestHeaderAllowedNames)
		if err != nil {
			utilruntime.HandleError(err)
			return
		}
	}

	if err := writeConfigMap(client, "extension-apiserver-authentication", data); err != nil {
		utilruntime.HandleError(err)
	}

	return
}

func jsonSerializeStringSlice(in []string) (string, error) {
	out, err := json.Marshal(in)
	if err != nil {
		return "", err
	}
	return string(out), err
}

func writeConfigMap(client coreclient.ConfigMapsGetter, name string, data map[string]string) error {
	existing, err := client.ConfigMaps(metav1.NamespaceSystem).Get(name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		_, err := client.ConfigMaps(metav1.NamespaceSystem).Create(&api.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Namespace: metav1.NamespaceSystem, Name: name},
			Data:       data,
		})
		return err
	}
	if err != nil {
		return err
	}

	existing.Data = data
	_, err = client.ConfigMaps(metav1.NamespaceSystem).Update(existing)
	return err
}
