/*
Copyright 2015 The Kubernetes Authors.

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

// Package configz serves ComponentConfig objects from running components.
//
// Each component that wants to serve its ComponentConfig creates a Config
// object, and the program should call InstallHandler once. e.g.,
//
//	func main() {
//		if err := setupConfigz(); err != nil {
//			klog.Fatalf("Failed to setup configz: %v", err)
//		}
//		http.ListenAndServe(":8080", http.DefaultServeMux)
//	}
//
//	func setupConfigz() error {
//		boatConfig := getBoatConfig()
//		planeConfig := getPlaneConfig()
//
//		bcz, err := configz.New("boat")
//		if err != nil {
//			return fmt.Errorf("unable to register boat config: %w", err)
//		}
//		if err := bcz.Set(boatConfig); err != nil {
//			return fmt.Errorf("unable to set boat config: %w", err)
//		}
//
//		pcz, err := configz.New("plane")
//		if err != nil {
//			return fmt.Errorf("unable to register plane config: %w", err)
//		}
//		if err := pcz.Set(planeConfig); err != nil {
//			return fmt.Errorf("unable to set plane config: %w", err)
//		}
//
//		configz.InstallHandler(http.DefaultServeMux)
//		return nil
//	}
package configz

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"k8s.io/apimachinery/pkg/runtime"
)

const DefaultConfigzPath = "/configz"

var (
	configsGuard sync.RWMutex
	configs      = map[string]*Config{}
)

// Config is a handle to a ComponentConfig object. Don't create these directly;
// use New() instead.
type Config struct {
	val runtime.Object
}

// InstallHandler adds an HTTP handler on the given mux for the "/configz"
// endpoint which serves all registered ComponentConfigs in JSON format.
func InstallHandler(m mux) {
	m.Handle(DefaultConfigzPath, http.HandlerFunc(handle))
}

type mux interface {
	Handle(string, http.Handler)
}

// New creates a Config object with the given name. Each Config is registered
// with this package's "/configz" handler.
func New(name string) (*Config, error) {
	configsGuard.Lock()
	defer configsGuard.Unlock()
	if _, found := configs[name]; found {
		return nil, fmt.Errorf("register config %q twice", name)
	}
	newConfig := Config{}
	configs[name] = &newConfig
	return &newConfig, nil
}

// Delete removes the named ComponentConfig from this package's "/configz"
// handler.
func Delete(name string) {
	configsGuard.Lock()
	defer configsGuard.Unlock()
	delete(configs, name)
}

// Set sets the ComponentConfig for this Config.
func (v *Config) Set(val runtime.Object) error {
	configsGuard.Lock()
	defer configsGuard.Unlock()
	if val == nil {
		return fmt.Errorf("val may not be nil")
	}
	gvk := val.GetObjectKind().GroupVersionKind()
	if len(gvk.Kind) == 0 {
		return fmt.Errorf("val must specify a kind")
	}
	if gvk.GroupVersion().Empty() {
		return fmt.Errorf("val must specify a group/version")
	}
	if gvk.Version == runtime.APIVersionInternal {
		return fmt.Errorf("val must specify an external version")
	}
	v.val = val
	return nil
}

// MarshalJSON marshals the ComponentConfig as JSON data.
func (v *Config) MarshalJSON() ([]byte, error) {
	return json.Marshal(v.val)
}

func handle(w http.ResponseWriter, r *http.Request) {
	if err := write(w); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func write(w http.ResponseWriter) error {
	var b []byte
	var err error
	func() {
		configsGuard.RLock()
		defer configsGuard.RUnlock()
		b, err = json.Marshal(configs)
	}()
	if err != nil {
		return fmt.Errorf("error marshaling json: %v", err)
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	_, err = w.Write(b)
	return err
}
