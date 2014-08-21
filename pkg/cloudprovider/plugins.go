/*
Copyright 2014 Google Inc. All rights reserved.

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

package cloudprovider

import (
	"sync"

	"github.com/golang/glog"
)

// Factory is a function that returns a cloudprovider.Interface.
type Factory func() (Interface, error)

// All registered cloud providers.
var providersMutex sync.Mutex
var providers = make(map[string]Factory)

// RegisterCloudProvider registers a cloudprovider.Factory by name.  This
// is expected to happen during app startup.
func RegisterCloudProvider(name string, cloud Factory) {
	providersMutex.Lock()
	defer providersMutex.Unlock()
	_, found := providers[name]
	if found {
		glog.Fatalf("Cloud provider %q was registered twice", name)
	}
	glog.Infof("Registered cloud provider %q", name)
	providers[name] = cloud
}

// GetCloudProvider creates an instance of the named cloud provider, or nil if
// the name is not known.  The error return is only used if the named provider
// was known but failed to initialize.
func GetCloudProvider(name string) (Interface, error) {
	providersMutex.Lock()
	defer providersMutex.Unlock()
	f, found := providers[name]
	if !found {
		return nil, nil
	}
	return f()
}
