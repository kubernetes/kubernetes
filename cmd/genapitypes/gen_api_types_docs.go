/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/cmd/genutils"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/registered"
	"github.com/emicklei/go-restful/swagger"
	"github.com/golang/glog"
)

func main() {
	flag.Parse()
	out_path := "docs/api-types/"
	outDir, err := genutils.OutDir(out_path)
	if err != nil {
		glog.Fatalf("failed to get output directory: %v", err)
	}

	in_path := "api/swagger-spec/"
	inDir, err := filepath.Abs(in_path)
	if err != nil {
		glog.Fatalf("failed to get input directory: %v", err)
	}
	gen_apitypes_doc(outDir, inDir)
}

func gen_apitypes_doc(outDir, inDir string) {
	count := 0
	for _, version := range registered.RegisteredVersions {
		filename := inDir + "/" + version + ".json"
		glog.Infof("generate log for version %s, source file %s", version, filename)
		file, err := ioutil.ReadFile(filename)
		if err != nil {
			fmt.Fprintf(os.Stderr, "unexpected err: %v\n", err)
		}
		outDir = outDir + version + "/"
		os.MkdirAll(outDir, 0755)
		c, err := gen_apitype_doc_for_version(outDir, file)
		if err != nil {
			glog.Fatalf("error when parsing %s: %v\n", filename, err)
		}
		count += c
		glog.Infof("generated %d docs in %s", count, outDir)
	}
}

func findTopLevelObjects(file []byte) map[string]struct{} {
	apiDeclaration := swagger.ApiDeclaration{}
	if err := json.Unmarshal(file, &apiDeclaration); err != nil {
		glog.Fatal(err)
	}
	apiVersion := apiDeclaration.ApiVersion + "."

	subSet := make(map[string]struct{})
	topSet := make(map[string]struct{})

	for _, m := range apiDeclaration.Models.List {
		modelName := strings.TrimLeft(m.Name, apiVersion)
		if _, ok := subSet[modelName]; ok {
			continue
		}
		topSet[modelName] = struct{}{}
		for _, p := range m.Model.Properties.List {
			if p.Property.DataTypeFields.Type != nil {
				if *p.Property.DataTypeFields.Type == "array" {
					if p.Property.DataTypeFields.Items.Ref != nil {
						//don't add the ref into subSet if the model is for a list for a top level object, like PodList
						if (m.Name[len(m.Name)-4:] == "
						printRef(buf, p.Property.DataTypeFields.Items.Ref, apiVersion)
					} else if p.Property.DataTypeFields.Items.Type != nil {
						fmt.Fprintf(buf, "[]"+*p.Property.DataTypeFields.Items.Type)
					}
				} else {
					fmt.Fprintf(buf, *p.Property.DataTypeFields.Type)
				}
			} else {
				printRef(buf, p.Property.DataTypeFields.Ref, apiVersion)
			}
			fmt.Fprintf(buf, "\n  * **_description_**: "+p.Property.Description+"\n")
		}

	}
}

func gen_apitype_doc_for_version(outDir string, file []byte) (int, error) {
	apiDeclaration := swagger.ApiDeclaration{}
	if err := json.Unmarshal(file, &apiDeclaration); err != nil {
		glog.Fatal(err)
	}
	apiVersion := apiDeclaration.ApiVersion + "."

	refSet := make(map[string]struct{})

	for _, m := range apiDeclaration.Models.List {
		modelName := strings.TrimLeft(m.Name, apiVersion)
		if _, ok := refSet[modelName]; ok {
			continue
		}
		buf := new(bytes.Buffer)
		fmt.Fprintf(buf, "###"+strings.TrimLeft(m.Name, apiVersion)+"###\n")
		if len(m.Model.Description) > 0 {
			fmt.Fprintf(buf, "Type Description:"+m.Model.Description+"\n")
		}
		fmt.Fprintf(buf, "\n---\n")
		sort.Sort(byPropertyName(m.Model.Properties.List))
		for _, p := range m.Model.Properties.List {
			fmt.Fprintf(buf, "* "+p.Name+": ")
			fmt.Fprintf(buf, "\n  * **_type_**: ")
			if p.Property.DataTypeFields.Type != nil {
				if *p.Property.DataTypeFields.Type == "array" {
					if p.Property.DataTypeFields.Items.Ref != nil {
						fmt.Fprintf(buf, "[]")
						printRef(buf, p.Property.DataTypeFields.Items.Ref, apiVersion)
					} else if p.Property.DataTypeFields.Items.Type != nil {
						fmt.Fprintf(buf, "[]"+*p.Property.DataTypeFields.Items.Type)
					}
				} else {
					fmt.Fprintf(buf, *p.Property.DataTypeFields.Type)
				}
			} else {
				printRef(buf, p.Property.DataTypeFields.Ref, apiVersion)
			}
			fmt.Fprintf(buf, "\n  * **_description_**: "+p.Property.Description+"\n")
		}
		outFile, err := os.Create(outDir + strings.TrimLeft(m.Name, apiVersion) + ".md")
		if err != nil {
			glog.Fatal(err)
		}
		defer outFile.Close()
		if _, err = outFile.Write(buf.Bytes()); err != nil {
			glog.Fatal(err)
		}
	}
	return len(apiDeclaration.Models.List), nil
}

func printRef(buf *bytes.Buffer, ref *string, v string) {
	fmt.Fprintf(buf, "["+strings.TrimLeft(*ref, v)+"]"+"("+strings.TrimLeft(*ref, v)+".md"+")")
}

type byPropertyName []swagger.NamedModelProperty

func (v byPropertyName) Len() int           { return len(v) }
func (v byPropertyName) Swap(i, j int)      { v[i], v[j] = v[j], v[i] }
func (v byPropertyName) Less(i, j int) bool { return v[i].Name < v[j].Name }
