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

package cmd

import (
	"fmt"
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/config"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/ghodss/yaml"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

// DataToObjects converts the raw JSON data into API objects
func DataToObjects(m meta.RESTMapper, t runtime.ObjectTyper, data []byte) (result []runtime.Object, errors []error) {
	configObj := []runtime.RawExtension{}

	if err := yaml.Unmarshal(data, &configObj); err != nil {
		errors = append(errors, fmt.Errorf("config unmarshal: %v", err))
		return result, errors
	}

	for i, in := range configObj {
		version, kind, err := t.DataVersionAndKind(in.RawJSON)
		if err != nil {
			errors = append(errors, fmt.Errorf("item[%d] kind: %v", i, err))
			continue
		}

		mapping, err := m.RESTMapping(kind, version)
		if err != nil {
			errors = append(errors, fmt.Errorf("item[%d] mapping: %v", i, err))
			continue
		}

		obj, err := mapping.Codec.Decode(in.RawJSON)
		if err != nil {
			errors = append(errors, fmt.Errorf("item[%d] decode: %v", i, err))
			continue
		}
		result = append(result, obj)
	}
	return
}

func (f *Factory) NewCmdCreateAll(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "createall [-d directory] [-f filename]",
		Short: "Create all resources specified in a directory, filename or stdin",
		Long: `Create all resources contained in JSON file specified in a directory, filename or stdin

JSON and YAML formats are accepted.

Examples:
  $ kubectl createall -d configs/
  <creates all resources listed in JSON or YAML files, found recursively under the configs directory>

  $ kubectl createall -f config.json
  <creates all resources listed in config.json>

  $ cat config.json | kubectl apply -f -
  <creates all resources listed in config.json>`,
		Run: func(cmd *cobra.Command, args []string) {
			clientFunc := func(mapper *meta.RESTMapping) (config.RESTClientPoster, error) {
				client, err := f.RESTClient(cmd, mapper)
				checkErr(err)
				return client, nil
			}

			filename := GetFlagString(cmd, "filename")
			directory := GetFlagString(cmd, "directory")
			if (len(filename) == 0 && len(directory) == 0) || (len(filename) != 0 && len(directory) != 0) {
				usageError(cmd, "Must pass a directory or filename to update")
			}

			files := []string{}
			if len(filename) != 0 {
				files = append(files, filename)

			} else {
				files = append(GetFilesFromDir(directory, ".json"), GetFilesFromDir(directory, ".yaml")...)
			}

			for _, filename := range files {
				data, err := ReadConfigData(filename)
				checkErr(err)

				items, errs := DataToObjects(f.Mapper, f.Typer, data)
				applyErrs := config.CreateObjects(f.Typer, f.Mapper, clientFunc, items)

				errs = append(errs, applyErrs...)
				if len(errs) > 0 {
					for _, e := range errs {
						glog.Error(e)
					}
				}
			}
		},
	}
	cmd.Flags().StringP("directory", "d", "", "Directory of JSON or YAML files to use to update the resource")
	cmd.Flags().StringP("filename", "f", "", "Filename or URL to file to use to update the resource")
	return cmd
}
