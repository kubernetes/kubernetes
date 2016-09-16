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

package framework

//go:generate ../../../hack/update-bindata.sh

// See https://github.com/kubernetes/kubernetes/issues/23987
import "k8s.io/kubernetes/test/e2e/generated"

/*
ReadOrDie reads a file from gobindata.
Relies heavily on the successful generation of build artifacts as per the go:generate directives above.
*/
func ReadOrDie(filePath string) []byte {

	fileBytes, err := generated.Asset(filePath)
	if err != nil {
		gobindata_msg := "An error occured, possibly gobindata doesn't know about the file you're opening. For questions on maintaining gobindata, contact the sig-testing group."
		Logf("Available gobindata files: %v ", generated.AssetNames())
		Failf("Failed opening %v , with error %v.  %v.", filePath, err, gobindata_msg)
	}
	return fileBytes
}
