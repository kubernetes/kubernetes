/*
Copyright 2019 The Kubernetes Authors.

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

package networkproxy

import (
	"fmt"
	"io/ioutil"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/apis/apiserver"
)

// ReadNetworkProxyConfiguration reads the network proxy configuration at the specified path.
// It returns the loaded network proxy configuration if the input file aligns with the required syntax.
// If it does not align with the provided syntax, it returns a default configuration which should function as a no-op.
// It does this to preserve backward compatibility when there was no network proxy configuration.
// It returns an error if the file did not exist.
func ReadNetworkProxyConfiguration(configFilePath string, configScheme *runtime.Scheme) (*apiserver.ConnectivityServiceConfiguration, error) {
	if configFilePath == "" {
		return nil, nil
	}
	// a file was provided, so we just read it.
	data, err := ioutil.ReadFile(configFilePath)
	if err != nil {
		return nil, fmt.Errorf("unable to read network proxy configuration from %q [%v]", configFilePath, err)
	}
	codecs := serializer.NewCodecFactory(configScheme)
	decoder := codecs.UniversalDecoder()
	decodedObj, err := runtime.Decode(decoder, data)
	if err != nil {
		// we got an error where the decode wasn't related to a missing type
		if !(runtime.IsMissingVersion(err) || runtime.IsMissingKind(err) || runtime.IsNotRegisteredError(err)) {
			return nil, err
		}
		// TODO: ChefTako generate reasonable defaults.
		return nil, err
	}
	decodedConfig, ok := decodedObj.(*apiserver.ConnectivityServiceConfiguration)
	if !ok {
		return nil, fmt.Errorf("unexpected type: %T", decodedObj)
	}

	return decodedConfig, nil
}