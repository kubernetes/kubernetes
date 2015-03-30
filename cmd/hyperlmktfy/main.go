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

// A binary that can morph into all of the other lmktfy binaries. You can
// also soft-link to it busybox style.
package main

import (
	"os"
)

func main() {
	hk := HyperLMKTFY{
		Name: "hyperlmktfy",
		Long: "This is an all-in-one binary that can run any of the various LMKTFY servers.",
	}

	hk.AddServer(NewLMKTFYAPIServer())
	hk.AddServer(NewLMKTFYControllerManager())
	hk.AddServer(NewScheduler())
	hk.AddServer(NewLMKTFYlet())
	hk.AddServer(NewLMKTFYProxy())

	hk.RunToExit(os.Args)
}
