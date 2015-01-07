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

package clientcmd

import (
	"fmt"
	"io/ioutil"
	"os"

	"gopkg.in/v2/yaml"
)

var (
	testConfigAlfa = Config{
		AuthInfos: map[string]AuthInfo{
			"red-user": {Token: "red-token"}},
		Clusters: map[string]Cluster{
			"cow-cluster": {Server: "http://cow.org:8080"}},
		Contexts: map[string]Context{
			"federal-context": {AuthInfo: "red-user", Cluster: "cow-cluster", Namespace: "hammer-ns"}},
	}
	testConfigBravo = Config{
		AuthInfos: map[string]AuthInfo{
			"black-user": {Token: "black-token"}},
		Clusters: map[string]Cluster{
			"pig-cluster": {Server: "http://pig.org:8080"}},
		Contexts: map[string]Context{
			"queen-anne-context": {AuthInfo: "black-user", Cluster: "pig-cluster", Namespace: "saw-ns"}},
	}
	testConfigCharlie = Config{
		AuthInfos: map[string]AuthInfo{
			"green-user": {Token: "green-token"}},
		Clusters: map[string]Cluster{
			"horse-cluster": {Server: "http://horse.org:8080"}},
		Contexts: map[string]Context{
			"shaker-context": {AuthInfo: "green-user", Cluster: "horse-cluster", Namespace: "chisel-ns"}},
	}
	testConfigDelta = Config{
		AuthInfos: map[string]AuthInfo{
			"blue-user": {Token: "blue-token"}},
		Clusters: map[string]Cluster{
			"chicken-cluster": {Server: "http://chicken.org:8080"}},
		Contexts: map[string]Context{
			"gothic-context": {AuthInfo: "blue-user", Cluster: "chicken-cluster", Namespace: "plane-ns"}},
	}
	testConfigConflictAlfa = Config{
		AuthInfos: map[string]AuthInfo{
			"red-user":    {Token: "a-different-red-token"},
			"yellow-user": {Token: "yellow-token"}},
		Clusters: map[string]Cluster{
			"cow-cluster":    {Server: "http://a-different-cow.org:8080", InsecureSkipTLSVerify: true},
			"donkey-cluster": {Server: "http://donkey.org:8080", InsecureSkipTLSVerify: true}},
		CurrentContext: "federal-context",
	}
)

func ExampleMergingSomeWithConflict() {
	commandLineFile, _ := ioutil.TempFile("", "")
	defer os.Remove(commandLineFile.Name())
	envVarFile, _ := ioutil.TempFile("", "")
	defer os.Remove(envVarFile.Name())

	WriteToFile(testConfigAlfa, commandLineFile.Name())
	WriteToFile(testConfigConflictAlfa, envVarFile.Name())

	loadingRules := ClientConfigLoadingRules{
		CommandLinePath: commandLineFile.Name(),
		EnvVarPath:      envVarFile.Name(),
	}

	mergedConfig, err := loadingRules.Load()

	output, err := yaml.Marshal(mergedConfig)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}

	fmt.Printf("%v", string(output))
	// Output:
	// preferences: {}
	// clusters:
	//   cow-cluster:
	//     server: http://cow.org:8080
	//   donkey-cluster:
	//     server: http://donkey.org:8080
	//     insecure-skip-tls-verify: true
	// users:
	//   red-user:
	//     token: red-token
	//   yellow-user:
	//     token: yellow-token
	// contexts:
	//   federal-context:
	//     cluster: cow-cluster
	//     user: red-user
	//     namespace: hammer-ns
	// current-context: federal-context
}

func ExampleMergingEverythingNoConflicts() {
	commandLineFile, _ := ioutil.TempFile("", "")
	defer os.Remove(commandLineFile.Name())
	envVarFile, _ := ioutil.TempFile("", "")
	defer os.Remove(envVarFile.Name())
	currentDirFile, _ := ioutil.TempFile("", "")
	defer os.Remove(currentDirFile.Name())
	homeDirFile, _ := ioutil.TempFile("", "")
	defer os.Remove(homeDirFile.Name())

	WriteToFile(testConfigAlfa, commandLineFile.Name())
	WriteToFile(testConfigBravo, envVarFile.Name())
	WriteToFile(testConfigCharlie, currentDirFile.Name())
	WriteToFile(testConfigDelta, homeDirFile.Name())

	loadingRules := ClientConfigLoadingRules{
		CommandLinePath:      commandLineFile.Name(),
		EnvVarPath:           envVarFile.Name(),
		CurrentDirectoryPath: currentDirFile.Name(),
		HomeDirectoryPath:    homeDirFile.Name(),
	}

	mergedConfig, err := loadingRules.Load()

	output, err := yaml.Marshal(mergedConfig)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}

	fmt.Printf("%v", string(output))
	// Output:
	// preferences: {}
	// clusters:
	//   chicken-cluster:
	//     server: http://chicken.org:8080
	//   cow-cluster:
	//     server: http://cow.org:8080
	//   horse-cluster:
	//     server: http://horse.org:8080
	//   pig-cluster:
	//     server: http://pig.org:8080
	// users:
	//   black-user:
	//     token: black-token
	//   blue-user:
	//     token: blue-token
	//   green-user:
	//     token: green-token
	//   red-user:
	//     token: red-token
	// contexts:
	//   federal-context:
	//     cluster: cow-cluster
	//     user: red-user
	//     namespace: hammer-ns
	//   gothic-context:
	//     cluster: chicken-cluster
	//     user: blue-user
	//     namespace: plane-ns
	//   queen-anne-context:
	//     cluster: pig-cluster
	//     user: black-user
	//     namespace: saw-ns
	//   shaker-context:
	//     cluster: horse-cluster
	//     user: green-user
	//     namespace: chisel-ns
	// current-context: ""
}
