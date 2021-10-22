// Copyright 2018 Microsoft Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"github.com/Azure/azure-sdk-for-go/tools/apidiff/cmd"
)

// breaking changes definitions
//
// const
//
// 1. removal/renaming of a const
// 2. changing the type of a const
//
// func/method
//
// 1. removal/renaming of a func
// 2. addition/removal of params and/or return values
// 3. changing param/retval types
// 4. changing receiver/param/retval byval/byref
//
// struct
// 1. removal/renaming of a field
// 2. chaning a field's type
// 3. changing a field from anonymous to explicit or explicit to anonymous
// 4. changing byval/byref
//

func main() {
	cmd.Execute()
}
