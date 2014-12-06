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
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

// RESTCallItem contains temporary info to execute REST call
type RESTCallItem struct {
	Mapping   *meta.RESTMapping
	Client    kubectl.RESTClient
	Name      string
	Namespace string
}

func NewRESTCallItem(mapping *meta.RESTMapping, client kubectl.RESTClient, namespace, name string) *RESTCallItem {
	return &RESTCallItem{
		Mapping:   mapping,
		Client:    client,
		Name:      name,
		Namespace: namespace,
	}
}

func (f *Factory) NewCmdDelete(out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "delete ([-f filename] | [<resource type> [, <resource_type>]* -l labelSelector] | [<resource> <id>])",
		Short: "Delete a resource by filename, stdin. Comma separted resource types and label selector, or  resource and id",
		Long: `Delete a resource by filename, stdin. Comma separated reosurce types and label selector or resource and id.

JSON and YAML formats are accepted.

If both a filename and command line arguments are passed, the command line
arguments are used and the filename is ignored.

Note that the delete command does NOT do resource version checks, so if someone
submits an update to a resource right when you submit a delete, their update
will be lost along with the rest of the resource.

Examples: l
  $ kubectl delete -f pod.json
  <delete a pod using the type and id pod.json>

  $ cat pod.json | kubectl delete -f -
  <delete a pod based on the type and id in the json passed into stdin>

  $ kubectls delete pods,replicationControllers -l name=myLabel
  <delete all pods and replication controllers with label name=myLabel>

  $ kubectl delete all -l name=myLabel
  <delete all pods, replication controllers, services, minions and events with label name=myLabel>

  $ kubectl delete pod 1234-56-7890-234234-456456
  <delete a pod with ID 1234-56-7890-234234-456456>`,
		Run: func(cmd *cobra.Command, args []string) {

			selector := GetFlagString(cmd, "selector")
			filename := GetFlagString(cmd, "filename")

			if len(selector) > 0 && len(filename) > 0 {
				usageError(cmd, "Cannot specify filename and label selector")
			}

			var RESTCallData []RESTCallItem

			if len(filename) > 0 && len(args) == 0 {
				mapping, namespace, name, _ := ResourceFromFile(filename, f.Typer, f.Mapper)
				client, err := f.Client(cmd, mapping)
				checkErr(err)
				RESTCallData = append(RESTCallData, *NewRESTCallItem(mapping, client, namespace, name))
			}

			if len(filename) == 0 {
				extractedArgs := ExtractAndExpandCommaSeparatedArgs(args)
				if len(selector) == 0 && len(extractedArgs) == 2 { // to delete resource by id
					mapping, namespace, name := ResourceFromArgs(cmd, extractedArgs, f.Mapper)
					client, err := f.Client(cmd, mapping)
					checkErr(err)
					RESTCallData = append(RESTCallData, *NewRESTCallItem(mapping, client, namespace, name))
				} else if len(selector) > 0 { // gather resources with specific labels
					namespace := getKubeNamespace(cmd)
					for _, arg := range extractedArgs {
						resource := kubectl.ExpandResourceShortcut(arg)
						if len(resource) == 0 {
							usageError(cmd, "Unknown resource %s", resource)
						}
						version, kind, err := f.Mapper.VersionAndKindForResource(resource)
						checkErr(err)
						mapping, err := f.Mapper.RESTMapping(version, kind)
						checkErr(err)
						client, err := f.Client(cmd, mapping)
						checkErr(err)
						labelSelector, err := labels.ParseSelector(selector)
						checkErr(err)
						restHelper := kubectl.NewRESTHelper(client, mapping)
						obj, err := restHelper.Get(namespace, "", labelSelector)
						checkErr(err)
						var objectsName []string
						extractNames(obj, &objectsName)
						for _, objectName := range objectsName {
							RESTCallData = append(RESTCallData, *NewRESTCallItem(mapping, client, namespace, objectName))
						}
					}
				}
			}

			errs := util.ErrorList{}
			for _, r := range RESTCallData {
				err := kubectl.NewRESTHelper(r.Client, r.Mapping).Delete(r.Namespace, r.Name)
				if err != nil {
					errs = append(errs, fmt.Errorf("Unable to delete %s. %v", r.Name, err))
					continue
				}
				fmt.Fprintf(out, "%s\n", r.Name)
			}

			for _, e := range errs {
				glog.Error(e)
			}

		},
	}
	cmd.Flags().StringP("filename", "f", "", "Filename or URL to file to use to delete the resource")
	cmd.Flags().StringP("selector", "l", "", "Selector (label query) to filter on")
	return cmd
}

//
func validateExtractNameFunction(extractFuncValue reflect.Value) error {
	if extractFuncValue.Kind() != reflect.Func {
		return fmt.Errorf("Unable to add to map. %#v is not a function.", extractFuncValue)
	}

	funcType := extractFuncValue.Type()
	if funcType.NumIn() != 2 || funcType.NumOut() != 1 {
		return fmt.Errorf("Bad function signature: it must take 2 parameters and return 1 value")
	}

	if funcType.In(1) != reflect.TypeOf((**[]string)(nil)).Elem() ||
		funcType.Out(0) != reflect.TypeOf((*error)(nil)).Elem() {
		return fmt.Errorf("Bad function signature.")
	}
	return nil

}

func addToMap(m map[reflect.Type]interface{}, extractFunc interface{}) error {

	extractFuncValue := reflect.ValueOf(extractFunc)
	if err := validateExtractNameFunction(extractFuncValue); err != nil {
		glog.Errorf("Unable to add extract function: %v", err)
		return err
	}

	if _, present := m[extractFuncValue.Type().In(0)]; present {
		return fmt.Errorf("Function to extract from type %T already handled",
			extractFuncValue.Type().In(0))
	}
	m[extractFuncValue.Type().In(0)] = extractFunc
	return nil
}

func extractNames(obj runtime.Object, items *[]string) error {

	// populate map of extract function
	extractFuncs := make(map[reflect.Type]interface{})
	err := addToMap(extractFuncs, extractPodName)
	checkErr(err)
	err = addToMap(extractFuncs, extractRCName)
	checkErr(err)
	err = addToMap(extractFuncs, extractSvcName)
	checkErr(err)
	err = addToMap(extractFuncs, extractEventName)
	checkErr(err)
	err = addToMap(extractFuncs, extractMinionName)
	checkErr(err)

	var objs []runtime.Object
	if runtime.IsListType(obj) {
		objs, _ = runtime.ExtractList(obj)
	} else {
		objs = append(objs, obj)
	}

	for _, obj := range objs {
		if extractFunction, present := extractFuncs[reflect.TypeOf(obj)]; present {
			extractFuncValue := reflect.ValueOf(extractFunction)
			args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(items)}
			resultValue := extractFuncValue.Call(args)[0]
			if !resultValue.IsNil() {
				fmt.Errorf("Unable to extract value %s", err)
			}
		}
	}
	return nil
}

func extractPodName(pod *api.Pod, items *[]string) error {
	*items = append(*items, pod.Name)
	return nil
}

func extractRCName(controller *api.ReplicationController, items *[]string) error {
	*items = append(*items, controller.Name)
	return nil
}

func extractSvcName(svc *api.Service, items *[]string) error {
	*items = append(*items, svc.Name)
	return nil
}

func extractEventName(event *api.Event, items *[]string) error {
	*items = append(*items, event.Name)
	return nil
}

func extractMinionName(minion *api.Minion, items *[]string) error {
	*items = append(*items, minion.Name)
	return nil
}
