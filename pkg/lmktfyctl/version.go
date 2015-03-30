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

package lmktfyctl

import (
	"fmt"
	"io"
	"os"

	"github.com/GoogleCloudPlatform/lmktfy/pkg/client"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/version"
)

func GetVersion(w io.Writer, lmktfyClient client.Interface) {
	GetClientVersion(w)

	serverVersion, err := lmktfyClient.ServerVersion()
	if err != nil {
		fmt.Printf("Couldn't read version from server: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(w, "Server Version: %#v\n", *serverVersion)
}

func GetClientVersion(w io.Writer) {
	fmt.Fprintf(w, "Client Version: %#v\n", version.Get())
}

func GetApiVersions(w io.Writer, lmktfyClient client.Interface) {
	apiVersions, err := lmktfyClient.ServerAPIVersions()
	if err != nil {
		fmt.Printf("Couldn't get available api versions from server: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(w, "Available Server Api Versions: %#v\n", *apiVersions)
}
