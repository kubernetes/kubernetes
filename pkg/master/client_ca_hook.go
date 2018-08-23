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
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	api "k8s.io/kubernetes/pkg/apis/core"
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
	// initializing CAs is important so that aggregated API servers can come up with "normal" config.
	// We've seen lagging etcd before, so we want to retry this a few times before we decide to crashloop
	// the API server on it.
	err := wait.Poll(1*time.Second, 30*time.Second, func() (done bool, err error) {
		// retry building the config since sometimes the server can be in an in-between state which caused
		// some kind of auto detection failure as I recall from other post start hooks.
		// TODO see if this is still true and fix the RBAC one too if it isn't.
		client, err := coreclient.NewForConfig(hookContext.LoopbackClientConfig)
		if err != nil {
			utilruntime.HandleError(err)
			return false, nil
		}

		return h.tryToWriteClientCAs(client)
	})

	// if we're never able to make it through initialization, kill the API server
	if err != nil {
		return fmt.Errorf("unable to initialize client CA configmap: %v", err)
	}

	return nil
}

// tryToWriteClientCAs is here for unit testing with a fake client.  This is a wait.ConditionFunc so the bool
// indicates if the condition was met.  True when its finished, false when it should retry.
func (h ClientCARegistrationHook) tryToWriteClientCAs(client coreclient.CoreInterface) (bool, error) {
	if err := createNamespaceIfNeeded(client, metav1.NamespaceSystem); err != nil {
		utilruntime.HandleError(err)
		return false, nil
	}

	data := map[string]string{}
	if len(h.ClientCA) > 0 {
		data["client-ca-file"] = string(h.ClientCA)
	}

	if len(h.RequestHeaderCA) > 0 {
		var err error

		// encoding errors aren't going to get better, so just fail on them.
		data["requestheader-username-headers"], err = jsonSerializeStringSlice(h.RequestHeaderUsernameHeaders)
		if err != nil {
			return false, err
		}
		data["requestheader-group-headers"], err = jsonSerializeStringSlice(h.RequestHeaderGroupHeaders)
		if err != nil {
			return false, err
		}
		data["requestheader-extra-headers-prefix"], err = jsonSerializeStringSlice(h.RequestHeaderExtraHeaderPrefixes)
		if err != nil {
			return false, err
		}
		data["requestheader-client-ca-file"] = string(h.RequestHeaderCA)
		data["requestheader-allowed-names"], err = jsonSerializeStringSlice(h.RequestHeaderAllowedNames)
		if err != nil {
			return false, err
		}
	}

	// write errors may work next time if we retry, so queue for retry
	if err := writeConfigMap(client, "extension-apiserver-authentication", data); err != nil {
		utilruntime.HandleError(err)
		return false, nil
	}

	return true, nil
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
