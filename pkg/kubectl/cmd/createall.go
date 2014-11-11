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
	"io"

	errs "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/config"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
	"gopkg.in/v1/yaml"
)

// DataToObjects converts the raw JSON data into API objects
func DataToObjects(m meta.RESTMapper, t runtime.ObjectTyper, data []byte) (result []runtime.Object, errors errs.ValidationErrorList) {
	configObj := []runtime.RawExtension{}

	if err := yaml.Unmarshal(data, &configObj); err != nil {
		errors = append(errors, errs.NewFieldInvalid("unmarshal", err))
		return result, errors.Prefix("Config")
	}

	for i, in := range configObj {
		version, kind, err := t.DataVersionAndKind(in.RawJSON)
		if err != nil {
			itemErrs := errs.ValidationErrorList{}
			itemErrs = append(itemErrs, errs.NewFieldInvalid("kind", string(in.RawJSON)))
			errors = append(errors, itemErrs.PrefixIndex(i).Prefix("item")...)
			continue
		}

		mapping, err := m.RESTMapping(version, kind)
		if err != nil {
			itemErrs := errs.ValidationErrorList{}
			itemErrs = append(itemErrs, errs.NewFieldRequired("mapping", err))
			errors = append(errors, itemErrs.PrefixIndex(i).Prefix("item")...)
			continue
		}

		obj, err := mapping.Codec.Decode(in.RawJSON)
		if err != nil {
			itemErrs := errs.ValidationErrorList{}
			itemErrs = append(itemErrs, errs.NewFieldInvalid("decode", err))
			errors = append(errors, itemErrs.PrefixIndex(i).Prefix("item")...)
			continue
		}
		result = append(result, obj)
	}
	return
}

func (f *Factory) NewCmdCreateAll(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "createall -f filename",
		Short: "Create all resources specified in filename or stdin",
		Long: `Create all resources contained in JSON file specified in filename or stdin

JSON and YAML formats are accepted.

Examples:
  $ kubectl createall -f config.json
  <creates all resources listed in config.json>

  $ cat config.json | kubectl apply -f -
  <creates all resources listed in config.json>`,
		Run: func(cmd *cobra.Command, args []string) {
			clientFunc := func(*meta.RESTMapping) (*client.RESTClient, error) {
				return getKubeClient(cmd).RESTClient, nil
			}

			filename := GetFlagString(cmd, "filename")
			if len(filename) == 0 {
				usageError(cmd, "Must pass a filename to update")
			}

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
		},
	}
	cmd.Flags().StringP("filename", "f", "", "Filename or URL to file to use to update the resource")
	return cmd
}
