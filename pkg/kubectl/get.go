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

package kubectl

import (
	"fmt"
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

func Get(w io.Writer, c *client.RESTClient, namespace string, resource string, id string, selector string, format string, noHeaders bool, templateFile string) error {
	path, err := resolveResource(resolveToPath, resource)
	if err != nil {
		return err
	}

	r := c.Verb("GET").Namespace(namespace).Path(path)
	if len(id) > 0 {
		r.Path(id)
	}
	if len(selector) > 0 {
		r.ParseSelectorParam("labels", selector)
	}
	result := r.Do()
	obj, err := result.Get()
	if err != nil {
		return err
	}

	printer, err := getPrinter(format, templateFile, noHeaders)
	if err != nil {
		return err
	}

	if err = printer.PrintObj(obj, w); err != nil {
		body, _ := result.Raw()
		return fmt.Errorf("Failed to print: %v\nRaw received object:\n%#v\n\nBody received: %v", err, obj, string(body))
	}

	return nil
}
