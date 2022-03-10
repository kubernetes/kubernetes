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

package main

import (
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	credentialproviderv1alpha1 "k8s.io/kubelet/pkg/apis/credentialprovider/v1alpha1"
)

func main() {
	if err := getCredentials(os.Stdout); err != nil {
		klog.Fatalf("failed to get credentials: %v", err)
	}
}

func getCredentials(w io.Writer) error {
	provider := &provider{
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}

	data, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		return err
	}

	var authRequest credentialproviderv1alpha1.CredentialProviderRequest
	err = json.Unmarshal(data, &authRequest)
	if err != nil {
		return err
	}

	auth, err := provider.Provide(authRequest.Image)
	if err != nil {
		return err
	}

	response := &credentialproviderv1alpha1.CredentialProviderResponse{
		TypeMeta: metav1.TypeMeta{
			Kind:       "CredentialProviderResponse",
			APIVersion: "credentialprovider.kubelet.k8s.io/v1alpha1",
		},
		CacheKeyType: credentialproviderv1alpha1.RegistryPluginCacheKeyType,
		Auth:         auth,
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		// The error from json.Marshal is intentionally not included so as to not leak credentials into the logs
		return errors.New("error marshaling response")
	}

	return nil
}
