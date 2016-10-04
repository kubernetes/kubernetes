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

package encryptionprovider

import (
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/golang/glog"
)

// Factory is a function that returns a encryptionprovider.Interface.
// The config parameter provides an io.Reader handler to the factory in
// order to load specific configurations. If no configuration is provided
// the parameter is nil.
type Factory func(config io.Reader) (Interface, error)

// All registered encryption providers.
var providersMutex sync.Mutex
var providers = make(map[string]Factory)

// RegisterEncryptionProvider registers a encryption provider.Factory by name.  This
// is expected to happen during app startup.
func RegisterEncryptionProvider(name string, encryptionProvider Factory) {
	providersMutex.Lock()
	defer providersMutex.Unlock()
	if _, found := providers[name]; found {
		glog.Fatalf("Encryption provider %q was registered twice", name)
	}
	glog.V(1).Infof("Registered encryption provider %q", name)
	providers[name] = encryptionProvider
}

// GetEncryptionProvider creates an instance of the named encryption provider, or nil if
// the name is not known.  The error return is only used if the named provider
// was known but failed to initialize. The config parameter specifies the
// io.Reader handler of the configuration file for the encryption provider, or nil
// for no configuation.
func GetEncryptionProvider(name string, config io.Reader) (Interface, error) {
	providersMutex.Lock()
	defer providersMutex.Unlock()
	f, found := providers[name]
	if !found {
		return nil, nil
	}

	return f(config)
}

// InitEncryptionProvider creates an instance of the named encryption provider.
func InitEncryptionProvider(name string, configFilePath string) (Interface, error) {
	var encryptionType Interface
	var err error

	if name == "" {
		glog.Info("No encryption provider specified.")
		return nil, nil
	}

	if configFilePath != "" {
		var config *os.File
		config, err = os.Open(configFilePath)
		if err != nil {
			glog.Fatalf("Couldn't open encryption provider configuration %s: %#v",
				configFilePath, err)
		}

		defer config.Close()
		encryptionType, err = GetEncryptionProvider(name, config)
	} else {
		// Pass explicit nil so plugins can actually check for nil. See
		// "Why is my nil error value not equal to nil?" in golang.org/doc/faq.
		encryptionType, err = GetEncryptionProvider(name, nil)
	}

	if err != nil {
		return nil, fmt.Errorf("could not init encryption provider %q: %v", name, err)
	}
	if encryptionType == nil {
		return nil, fmt.Errorf("unknown encryption provider %q", name)
	}

	return encryptionType, nil
}
