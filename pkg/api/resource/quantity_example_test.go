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

package resource_test

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
)

func ExampleFormat() {
	memorySize := resource.NewQuantity(5*1024*1024*1024, resource.BinarySI)
	fmt.Printf("memorySize = %v\n", memorySize)

	diskSize := resource.NewQuantity(5*1000*1000*1000, resource.DecimalSI)
	fmt.Printf("diskSize = %v\n", diskSize)

	cores := resource.NewMilliQuantity(5300, resource.DecimalSI)
	fmt.Printf("cores = %v\n", cores)

	// Output:
	// memorySize = 5Gi
	// diskSize = 5G
	// cores = 5300m
}

func ExampleMustParse() {
	memorySize := resource.MustParse("5Gi")
	fmt.Printf("memorySize = %v (%v)\n", memorySize.Value(), memorySize.Format)

	diskSize := resource.MustParse("5G")
	fmt.Printf("diskSize = %v (%v)\n", diskSize.Value(), diskSize.Format)

	cores := resource.MustParse("5300m")
	fmt.Printf("milliCores = %v (%v)\n", cores.MilliValue(), cores.Format)

	cores2 := resource.MustParse("5.4")
	fmt.Printf("milliCores = %v (%v)\n", cores2.MilliValue(), cores2.Format)

	// Output:
	// memorySize = 5368709120 (BinarySI)
	// diskSize = 5000000000 (DecimalSI)
	// milliCores = 5300 (DecimalSI)
	// milliCores = 5400 (DecimalSI)
}
