/*
Copyright 2017 The Kubernetes Authors.

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

package temptest

import (
	"errors"
	"fmt"
	"io"

	"k8s.io/utils/temp"
)

func TestedCode(dir temp.Directory) error {
	f, err := dir.NewFile("filename")
	if err != nil {
		return err
	}
	_, err = io.WriteString(f, "Bonjour!")
	if err != nil {
		return err
	}
	return dir.Delete()
}

func Example() {
	dir := FakeDir{}

	err := TestedCode(&dir)
	if err != nil {
		panic(err)
	}

	if dir.Deleted == false {
		panic(errors.New("Directory should have been deleted"))
	}

	if dir.Files["filename"] == nil {
		panic(errors.New(`"filename" should have been created`))
	}

	fmt.Println(dir.Files["filename"].Buffer.String())
	// Output: Bonjour!
}
