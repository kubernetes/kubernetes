/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

/*
Package authcfg defines a file format for holding authentication
information needed by clients of Kubernetes.  Typically,
a Kubernetes cluster will put auth info for the admin in a known
location when it is created, and will (soon) put it in a known
location within a Container's file tree for Containers that
need access to the Kubernetes API.

Having a defined format allows:
  - clients to be implmented in multiple languages
  - applications which link clients to be portable across
    clusters with different authentication styles (e.g.
    some may use SSL Client certs, others may not, etc)
  - when the format changes, applications only
    need to update this code.

The file format is json, marshalled from a struct authcfg.Info.

Clinet libraries in other languages should use the same format.

It is not intended to store general preferences, such as default
namespace, output options, etc.  CLIs (such as kubectl) and UIs should
develop their own format and may wish to inline the authcfg.Info type.

The authcfg.Info is just a file format.  It is distinct from
client.Config which holds options for creating a client.Client.
Helper functions are provided in this package to fill in a
client.Client from an authcfg.Info.

Example:

    import (
        "pkg/client"
        "pkg/clientauth"
    )

    info, err := clientauth.LoadFromFile(filename)
    if err != nil {
      // handle error
    }
    clientConfig = client.Config{}
    clientConfig.Host = "example.com:4901"
    clientConfig = info.MergeWithConfig()
    client := client.New(clientConfig)
    client.Pods(ns).List()
*/
package clientauth

// TODO: need a way to rotate Tokens.  Therefore, need a way for client object to be reset when the authcfg is updated.
import (
	"encoding/json"
	"io/ioutil"
	"os"

	"k8s.io/kubernetes/pkg/client"
)

// Info holds Kubernetes API authorization config.  It is intended
// to be read/written from a file as a JSON object.
type Info struct {
	User        string
	Password    string
	CAFile      string
	CertFile    string
	KeyFile     string
	BearerToken string
	Insecure    *bool
}

// LoadFromFile parses an Info object from a file path.
// If the file does not exist, then os.IsNotExist(err) == true
func LoadFromFile(path string) (*Info, error) {
	var info Info
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, err
	}
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(data, &info)
	if err != nil {
		return nil, err
	}
	return &info, err
}

// MergeWithConfig returns a copy of a client.Config with values from the Info.
// The fields of client.Config with a corresponding field in the Info are set
// with the value from the Info.
func (info Info) MergeWithConfig(c client.Config) (client.Config, error) {
	var config client.Config = c
	config.Username = info.User
	config.Password = info.Password
	config.CAFile = info.CAFile
	config.CertFile = info.CertFile
	config.KeyFile = info.KeyFile
	config.BearerToken = info.BearerToken
	if info.Insecure != nil {
		config.Insecure = *info.Insecure
	}
	return config, nil
}

func (info Info) Complete() bool {
	return len(info.User) > 0 ||
		len(info.CertFile) > 0 ||
		len(info.BearerToken) > 0
}
