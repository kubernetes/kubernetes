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

package main

import (
	"io"
	"net/http"
)

func hello(w http.ResponseWriter, r *http.Request) {
	io.WriteString(w, "I am a fake git server")

}

// When doing `git clone localhost:8000`, you will clone an empty git repo named "8000" on local.
// You can also use `git clone localhost:8000 my-repo-name` to rename that repo.
func main() {
	http.HandleFunc("/", hello)
	http.ListenAndServe(":8000", nil)
}
