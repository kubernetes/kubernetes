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

/*
This is a simple script that makes *every* environment variable available
as a go template field of the same name

$ echo "hello world, MYVAR={{.MYVAR}}" > test.txt
$ MYVAR=foobar go run template.go test.txt
> hello world, MYVAR=foobar

If you want the base64 version of any MYVAR, simple use {{.MYVAR_BASE64}}
*/

package main

import (
	"encoding/base64"
	"flag"
	"fmt"
	"io"
	"os"
	"path"
	"strings"
	"text/template"
)

func main() {
	flag.Parse()
	env := make(map[string]string)
	envList := os.Environ()

	for i := range envList {
		pieces := strings.SplitN(envList[i], "=", 2)
		if len(pieces) == 2 {
			env[pieces[0]] = pieces[1]
			env[pieces[0]+"_BASE64"] = base64.StdEncoding.EncodeToString([]byte(pieces[1]))
		} else {
			fmt.Fprintf(os.Stderr, "Invalid environ found: %s\n", envList[i])
			os.Exit(2)
		}
	}

	for i := 0; i < flag.NArg(); i++ {
		inpath := flag.Arg(i)

		if err := templateYamlFile(env, inpath, os.Stdout); err != nil {
			panic(err)
		}
	}
}

func templateYamlFile(params map[string]string, inpath string, out io.Writer) error {
	if tmpl, err := template.New(path.Base(inpath)).Option("missingkey=zero").ParseFiles(inpath); err != nil {
		return err
	} else {
		if err := tmpl.Execute(out, params); err != nil {
			return err
		}
	}
	_, err := out.Write([]byte("\n---\n"))
	return err
}
