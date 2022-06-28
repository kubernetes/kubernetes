/*
Copyright 2014 The Kubernetes Authors.

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

package config

import (
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	clientcmdapiv1 "k8s.io/client-go/tools/clientcmd/api/v1"
	"k8s.io/client-go/util/jsonpath"
	cliflag "k8s.io/component-base/cli/flag"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

type setOptions struct {
	configAccess  clientcmd.ConfigAccess
	propertyName  string
	propertyValue string
	setRawBytes   cliflag.Tristate
	jsonPath      bool
	deduplicate   bool
}

var (
	setLong = templates.LongDesc(i18n.T(`
	Set an individual value in a kubeconfig file.

	PROPERTY_NAME is either a jsonpath query for where the property name will be, or a dot delimited name.

	PROPERTY_VALUE is the new value you want to set. Binary fields such as 'certificate-authority-data' expect a base64 encoded string unless the --set-raw-bytes flag is used.

	Specifying an attribute name that already exists will merge new fields on top of existing values.`))

	setExample = templates.Examples(`
	# Set the server field on the my-cluster cluster to https://1.2.3.4
	kubectl config set clusters.my-cluster.server https://1.2.3.4

	# Set the certificate-authority-data field on the my-cluster cluster
	kubectl config set clusters.my-cluster.certificate-authority-data $(echo "cert_data_here" | base64 -i -)

	# Set the cluster field in the my-context context to my-cluster
	kubectl config set contexts.my-context.cluster my-cluster

	# Set the client-key-data field in the cluster-admin user using --set-raw-bytes option
	kubectl config set users.cluster-admin.client-key-data cert_data_here --set-raw-bytes=true

	# Set the server for a cluster with name cluster-0 using jsonpath
	kubectl config set '{.clusters[?(@.name=="cluster-0")].cluster.server}' "https://1.2.3.4"

	# Set the same username value for all users using the wildcard filter
	kubectl config set '{.users[*].user.username}' "test-user"

	# Set a new list using users[*].user.exec.args with jsonpath
	kubectl config set '{.users[*].user.exec.args}' "arg1,arg2"

	# Add a list item using users[*].user.exec.args with jsonpath
	kubectl config set '{.users[*].user.exec.args}' "arg3+"

	# Remove a list item using users[*].user.exec.args with jsonpath
	kubectl config set '{.users[*].user.exec.args}' "arg2-"

	# Set new list that will be deduplicated and sorted, will result in list of arg1,arg2,arg3,arg4
	kubectl config set '{.users[*].user.exec.args}' "arg1,arg2,agr2,arg4,arg3" --deduplicate

	# Deduplicate and sort existing list without making any updates to it
	kubectl config set '{.users[*].user.exec.args}' "+" --deduplicate`)
)

// NewCmdConfigSet returns a Command instance for 'config set' sub command
func NewCmdConfigSet(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &setOptions{configAccess: configAccess, jsonPath: false}

	cmd := &cobra.Command{
		Use:                   "set PROPERTY_NAME PROPERTY_VALUE",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Set an individual value in a kubeconfig file"),
		Long:                  setLong,
		Example:               setExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.complete(cmd))
			cmdutil.CheckErr(options.run())
			fmt.Fprintf(out, "Property %q set.\n", options.propertyName)
		},
	}

	f := cmd.Flags().VarPF(&options.setRawBytes, "set-raw-bytes", "", "When writing a []byte PROPERTY_VALUE, write the given string directly without base64 decoding.")
	f.NoOptDefVal = "true"
	cmd.Flags().BoolVar(&options.deduplicate, "deduplicate", false, "Whether to use deduplicate list of values or not. This flag will also sort the list.")

	return cmd
}

func (o setOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	if o.jsonPath {
		// Convert api config to apiv1 config, so we can use jsonpath properly
		v1Config := &clientcmdapiv1.Config{}
		if err := clientcmdapiv1.Convert_api_Config_To_v1_Config(config, v1Config, nil); err != nil {
			return err
		}

		if err := modifyConfigJson(v1Config, o.propertyName, o.propertyValue, false, o.setRawBytes.Value(), o.deduplicate); err != nil {
			return err
		}

		// Convert the apiv1 config back to an api config to write back out
		finalConfig := clientcmdapi.NewConfig()
		if err := clientcmdapiv1.Convert_v1_Config_To_api_Config(v1Config, finalConfig, nil); err != nil {
			return err
		}
		config = finalConfig
	} else {
		steps, err := newNavigationSteps(o.propertyName)
		if err != nil {
			return err
		}

		setRawBytes := false
		if o.setRawBytes.Provided() {
			setRawBytes = o.setRawBytes.Value()
		}

		err = modifyConfig(reflect.ValueOf(config), steps, o.propertyValue, false, setRawBytes)
		if err != nil {
			return err
		}
	}

	if err := clientcmd.ModifyConfig(o.configAccess, *config, false); err != nil {
		return err
	}

	return nil
}

func (o *setOptions) complete(cmd *cobra.Command) error {
	endingArgs := cmd.Flags().Args()
	if len(endingArgs) != 2 {
		return helpErrorf(cmd, "Unexpected args: %v", endingArgs)
	}

	o.propertyValue = endingArgs[1]
	o.propertyName = endingArgs[0]

	// try to determine if we have a jsonpath from first character of first argument
	// this should only ever be a { if this is a jsonpath string, if it is not we will try to use the old dot delimited
	// syntax instead.
	if string(endingArgs[0][0]) == "{" {
		o.jsonPath = true
	}
	return nil
}

func (o setOptions) validate() error {
	if len(o.propertyValue) == 0 {
		return errors.New("you cannot use set to unset a property")
	}

	if len(o.propertyName) == 0 {
		return errors.New("you must specify a property")
	}

	return nil
}

// modifyConfigJson: modifyConfig, but for JSON path
func modifyConfigJson(config *clientcmdapiv1.Config, propertyName, propertyValue string, unset bool, setRawBytes bool, deduplicate bool) error {
	// Create jsonpath parser to use throughout the function
	jsonPath := jsonpath.New("Value Getter").AllowMissingKeys(true)

	jsonPathParser, err := jsonpath.Parse("Node Parser", propertyName)
	if err != nil {
		return err
	}

	// Theoretically you can set as many outer node lists as you want in a jsonpath string
	// e.g. {.key1}{.key2}{.key3}...{.keyn}
	// so we're making sure we traverse all of them, even if it doesn't really make that much
	// sense that a user would pass a ton of key like that to be set to an identical value
	for _, outerNodeList := range jsonPathParser.Root.Nodes {
		var jsonPathTraverser []string
		innerNodeList := outerNodeList.(*jsonpath.ListNode).Nodes // The Root.Nodes type will always be a ListNode so this assertion is safe

		// This is where we enter the actual list of nodes that are contained within the {} in the jsonpath language
		// e.g. {.node1.node2.node3}
		for nodeIterator, node := range innerNodeList {
			// For set, it only makes sense to use field, array, and filter node types, so we will be ignoring
			// any other types of nodes and returning an error specifying what type of node was unsupported
			var filterKey string
			var filterValue string

			// Checking types so we can build the jsonPathTraverser accurately
			switch node.Type() {
			case jsonpath.NodeField:
				// Add . to field node values because they need to match the form .value
				jsonPathTraverser = append(jsonPathTraverser, "."+node.(*jsonpath.FieldNode).Value)

			case jsonpath.NodeArray:
				// Array nodes can just use their naked value
				jsonPathTraverser = append(jsonPathTraverser, node.(*jsonpath.ArrayNode).Value)

			case jsonpath.NodeFilter:
				filterNode := node.(*jsonpath.FilterNode)

				// If we're using the == operator for the filter, we will set the value if it is not found
				// We need to pull the key and value out to set them properly
				if filterNode.Operator == "==" {
					filterKey = filterNode.Left.Nodes[0].(*jsonpath.FieldNode).Value
					filterValue = filterNode.Right.Nodes[0].(*jsonpath.TextNode).Text
				}
				jsonPathTraverser = append(jsonPathTraverser, filterNode.Value)

			default:
				// At this point I don't think we need to support any of the other types of nodes
				// will reassess after initial PoC completed/implemented
				return fmt.Errorf("unsupported jsonpath node type detected: %v", jsonpath.NodeTypeName[node.Type()])
			}

			if err := jsonPath.Parse("{" + strings.Join(jsonPathTraverser, "") + "}"); err != nil {
				return err
			}
			results, err := jsonPath.FindResults(config)
			if err != nil {
				return err
			}

			// We must now work on the results gotten by our user provided jsonpath
			for _, outerResult := range results {
				// This should really only apply to the slices of named structs e.g. NamedClusters. It is a result of
				// filter returning an empty list in the results.
				if len(outerResult) == 0 && node.Type() == jsonpath.NodeFilter {
					// Need to back up so that we can create the new named struct type inside the outer list
					// This requires a new jsonpath object
					innerJsonPath := jsonpath.New("Inner Value Getter")
					if err := innerJsonPath.Parse("{" + strings.Join(jsonPathTraverser[0:nodeIterator], "") + "}"); err != nil {
						return err
					}
					newInnerResult, err := innerJsonPath.FindResults(config)
					if err != nil {
						return err
					}

					// This type must be a struct. If it is not something has gone wrong.
					if newInnerResult[0][0].Type().Elem().Kind() != reflect.Struct {
						return fmt.Errorf("filtered type is not a struct and must be, failed at path: %q", "{"+strings.Join(jsonPathTraverser[0:nodeIterator], "")+"}")
					}
					newType := reflect.New(newInnerResult[0][0].Type().Elem())

					// Set the new type's given field's value appropriately so that we can continue on down the node list
					// This will most likely be setting the "name" field
					fieldIndex := getStructFieldIndexByName(newType.Elem(), filterKey)
					newType.Elem().Field(fieldIndex).Set(reflect.ValueOf(filterValue))
					newInnerResult[0][0].Set(reflect.Append(newInnerResult[0][0], newType.Elem()))
				}

				// We now need to work on all the results in the inner results list
				for innerResultIndex, innerResult := range outerResult {
					// List of actions per result kind
					switch innerResult.Kind() {
					case reflect.Slice:
						// Need to check if we're working on a string slice or a slice of structs
						switch innerResult.Type().Elem().Kind() {
						case reflect.String:
							if unset {
								innerResult.Set(reflect.Zero(innerResult.Type()))
							} else if nodeIterator == len(innerNodeList)-1 && innerResult.CanAddr() {
								// We only use the custom slice setting function if this is the last node in the list
								// otherwise the next node in the list will find a different type, which we will use to
								// set the value
								currentSliceValue := innerResult.Interface().([]string)
								newSliceValue := editStringSlice(currentSliceValue, propertyValue, deduplicate)
								innerResult.Set(reflect.ValueOf(newSliceValue))
							}

						case reflect.Uint8:
							if nodeIterator < len(innerNodeList)-1 {
								return fmt.Errorf("can't have more nodes after a byte slice, %q", strings.Join(jsonPathTraverser[0:nodeIterator], ""))
							}
							if unset {
								innerResult.Set(reflect.Zero(innerResult.Type()))
							} else {
								if setRawBytes {
									innerResult.SetBytes([]byte(propertyValue))
								} else {
									val, err := base64.StdEncoding.DecodeString(propertyValue)
									if err != nil {
										return fmt.Errorf("error decoding input value: %v", err)
									}
									innerResult.SetBytes(val)
								}
							}

						case reflect.Struct:
							if unset && nodeIterator == len(innerNodeList)-1 {
								innerResult.Set(reflect.Zero(innerResult.Type()))
							} else {
								if nodeIterator == len(innerNodeList)-1 {
									return fmt.Errorf("struct node can not be the final node in the path: {%s}", strings.Join(jsonPathTraverser[0:nodeIterator+1], ""))
								}
							}

						default:
							return fmt.Errorf("can not set slice that is not a string, byte, or struct type")
						}

					case reflect.Map:
						switch innerResult.Type().Elem().Kind() {
						case reflect.String:
							// just pass in the property value provided by the user if we're working on a string
							if err := mapHandler(innerResult, innerNodeList, nodeIterator, reflect.ValueOf(propertyValue), unset); err != nil {
								return err
							}

						case reflect.Slice:
							// This isn't necessary now as there are no instances of any other map of slices that isn't
							// a map[string][]string, but just "for the future"
							if innerResult.Type().Elem().Elem().Kind() != reflect.String {
								return fmt.Errorf("can not set map value type that is not a string slice: {%s}", strings.Join(jsonPathTraverser[0:nodeIterator+1], ""))
							}
							// Set map if it isn't initialized
							if innerResult.IsNil() && !unset {
								innerResult.Set(reflect.MakeMap(innerResult.Type()))
							}

							if unset && nodeIterator == len(innerNodeList)-1 {
								innerResult.Set(reflect.Zero(innerResult.Type()))
							} else if unset && nodeIterator == len(innerNodeList)-2 {
								mapKey := innerNodeList[nodeIterator+1].(*jsonpath.FieldNode).Value
								innerResult.SetMapIndex(reflect.ValueOf(mapKey), reflect.Value{})
							} else if !unset {
								// Check and see if the key's value is zero or not and get current value if it exists
								var newSliceValue []string
								var currentSliceValue []string
								mapKey := innerNodeList[nodeIterator+1].(*jsonpath.FieldNode).Value
								mapValue := innerResult.MapIndex(reflect.ValueOf(mapKey))
								if mapValue.IsValid() {
									currentSliceValue = innerResult.MapIndex(reflect.ValueOf(mapKey)).Interface().([]string)
								}
								newSliceValue = editStringSlice(currentSliceValue, propertyValue, deduplicate)
								if err := mapHandler(innerResult, innerNodeList, nodeIterator, reflect.ValueOf(newSliceValue), unset); err != nil {
									return err
								}
							}
						}

					case reflect.String:
						if nodeIterator < len(innerNodeList)-1 {
							return fmt.Errorf("can't have more nodes after a string, {%s}", strings.Join(jsonPathTraverser[0:nodeIterator], ""))
						}
						if unset && innerResult.CanAddr() {
							innerResult.Set(reflect.Zero(innerResult.Type()))
						} else if innerResult.CanAddr() {
							innerResult.Set(reflect.ValueOf(propertyValue))
						}

					case reflect.Pointer:
						// Check to see if next filter value actually exists as a field for the given struct pointer
						nextNodeValue := innerNodeList[nodeIterator+1].(*jsonpath.FieldNode).Value
						innerPointerType := reflect.Zero(innerResult.Type().Elem())
						if fieldIndex := getStructFieldIndexByName(innerPointerType, nextNodeValue); fieldIndex == -1 {
							return fmt.Errorf("unable to parse path %q at %q", propertyName, nextNodeValue)
						}
						// We only need to do something here if the pointer is zero
						if innerResult.IsZero() {
							if err := pointerHandler(config, jsonPathTraverser, innerResultIndex, nodeIterator, propertyName); err != nil {
								return err
							}
						} else if nodeIterator == len(innerNodeList)-1 && unset {
							innerResult.Set(reflect.Zero(innerResult.Type()))
						}

					case reflect.Bool:
						if nodeIterator < len(innerNodeList)-1 {
							return fmt.Errorf("can't have more nodes after a bool, {%s}", strings.Join(jsonPathTraverser[0:nodeIterator], ""))
						}
						if unset {
							innerResult.Set(reflect.Zero(innerResult.Type()))
						}
						boolValue, err := toBool(propertyValue)
						if err != nil {
							return err
						}
						innerResult.SetBool(boolValue)

					case reflect.Struct:
						if unset && nodeIterator == len(innerNodeList)-1 {
							innerResult.Set(reflect.Zero(innerResult.Type()))
						} else if innerNodeList[nodeIterator+1].Type() != jsonpath.NodeField {
							return fmt.Errorf("next node after finding a struct must be a Field Node, error at %q", "{"+strings.Join(jsonPathTraverser[:nodeIterator], "")+"}")
						} else {
							// Checking that next field node value actually exists so that we can provide helpful error messages.
							nextNodeValue := innerNodeList[nodeIterator+1].(*jsonpath.FieldNode).Value
							if fieldIndex := getStructFieldIndexByName(innerResult, nextNodeValue); fieldIndex == -1 {
								return fmt.Errorf("unable to parse path %q at %q", propertyName, nextNodeValue)
							}
						}
					}
				}
			}
		}
	}

	return nil
}

func mapHandler(value reflect.Value, nodeList []jsonpath.Node, nodeIterator int, propertyValue reflect.Value, unset bool) error {
	// There should be one more node, if there is not or there is more than one more node return
	// a descriptive error.
	if nodeIterator == len(nodeList)-1 && !unset {
		return fmt.Errorf("must provide key when setting map, please check your jsonpath")
	}
	if nodeIterator <= len(nodeList)-3 {
		return fmt.Errorf("extra nodes detected beyond map's key node, please check your jsonpath")
	}

	if nodeIterator == len(nodeList)-1 && unset {
		value.Set(reflect.MakeMap(value.Type()))
		return nil
	}

	// Both key and value must be strings here and node bype must be field, fail if not
	keyKind := value.Type().Key().Kind()
	if !(keyKind == reflect.String) {
		return fmt.Errorf("failed to set map because it does not have string keys")
	}
	if nodeList[nodeIterator+1].Type() != jsonpath.NodeField {
		return fmt.Errorf("map key must be a field type node, e.g. \".key\"")
	}

	// safe type assertion because we're testing it above, safe slice position because we checked above
	mapKey := nodeList[nodeIterator+1].(*jsonpath.FieldNode).Value

	// Have to initialize map if it is nil
	if value.IsNil() && !unset {
		value.Set(reflect.MakeMap(value.Type()))
	}
	if value.IsNil() && unset {
		return nil
	}

	if unset {
		value.SetMapIndex(reflect.ValueOf(mapKey), reflect.Value{})
		return nil
	}
	value.SetMapIndex(reflect.ValueOf(mapKey), propertyValue)
	return nil
}

func pointerHandler(config *clientcmdapiv1.Config, jsonPathTraverser []string, innerResultIndex int, nodeIterator int, propertyName string) error {
	// Because we only work on pointers that are zero, we need to move back one on the stack and create the pointer
	// before we can continue on
	innerJsonPath := jsonpath.New("Inner Value Getter")
	if err := innerJsonPath.Parse("{" + strings.Join(jsonPathTraverser[0:nodeIterator], "") + "}"); err != nil {
		return err
	}
	newInnerResult, err := innerJsonPath.FindResults(config)
	if err != nil {
		return err
	}
	// because this a field node we need to remove the leading "." in the value
	fieldIndex := getStructFieldIndexByName(newInnerResult[0][innerResultIndex], jsonPathTraverser[nodeIterator][1:])
	if fieldIndex == -1 {
		return fmt.Errorf("could not find field %q in path %q", jsonPathTraverser[nodeIterator], propertyName)
	}
	fieldValue := newInnerResult[0][innerResultIndex].Field(fieldIndex)
	fieldValue.Set(reflect.New(fieldValue.Type().Elem()))

	return nil
}

func modifyConfig(curr reflect.Value, steps *navigationSteps, propertyValue string, unset bool, setRawBytes bool) error {
	currStep := steps.pop()

	actualCurrValue := curr
	if curr.Kind() == reflect.Pointer {
		actualCurrValue = curr.Elem()
	}

	switch actualCurrValue.Kind() {
	case reflect.Map:
		if !steps.moreStepsRemaining() && !unset {
			return fmt.Errorf("can't set a map to a value: %v", actualCurrValue)
		}

		mapKey := reflect.ValueOf(currStep.stepValue)
		mapValueType := curr.Type().Elem().Elem()

		if !steps.moreStepsRemaining() && unset {
			actualCurrValue.SetMapIndex(mapKey, reflect.Value{})
			return nil
		}

		currMapValue := actualCurrValue.MapIndex(mapKey)

		needToSetNewMapValue := currMapValue.Kind() == reflect.Invalid
		if needToSetNewMapValue {
			if unset {
				return fmt.Errorf("current map key `%v` is invalid", mapKey.Interface())
			}
			currMapValue = reflect.New(mapValueType.Elem()).Elem().Addr()
			actualCurrValue.SetMapIndex(mapKey, currMapValue)
		}

		err := modifyConfig(currMapValue, steps, propertyValue, unset, setRawBytes)
		if err != nil {
			return err
		}

		return nil

	case reflect.String:
		if steps.moreStepsRemaining() {
			return fmt.Errorf("can't have more steps after a string. %v", steps)
		}
		actualCurrValue.SetString(propertyValue)
		return nil

	case reflect.Slice:
		if steps.moreStepsRemaining() {
			return fmt.Errorf("can't have more steps after bytes. %v", steps)
		}
		innerKind := actualCurrValue.Type().Elem().Kind()
		if innerKind != reflect.Uint8 {
			return fmt.Errorf("unrecognized slice type. %v", innerKind)
		}

		if unset {
			actualCurrValue.Set(reflect.Zero(actualCurrValue.Type()))
			return nil
		}

		if setRawBytes {
			actualCurrValue.SetBytes([]byte(propertyValue))
		} else {
			val, err := base64.StdEncoding.DecodeString(propertyValue)
			if err != nil {
				return fmt.Errorf("error decoding input value: %v", err)
			}
			actualCurrValue.SetBytes(val)
		}
		return nil

	case reflect.Bool:
		if steps.moreStepsRemaining() {
			return fmt.Errorf("can't have more steps after a bool. %v", steps)
		}
		boolValue, err := toBool(propertyValue)
		if err != nil {
			return err
		}
		actualCurrValue.SetBool(boolValue)
		return nil

	case reflect.Struct:
		for fieldIndex := 0; fieldIndex < actualCurrValue.NumField(); fieldIndex++ {
			currFieldValue := actualCurrValue.Field(fieldIndex)
			currFieldType := actualCurrValue.Type().Field(fieldIndex)
			currYamlTag := currFieldType.Tag.Get("json")
			currFieldTypeYamlName := strings.Split(currYamlTag, ",")[0]

			if currFieldTypeYamlName == currStep.stepValue {
				thisMapHasNoValue := (currFieldValue.Kind() == reflect.Map && currFieldValue.IsNil())

				if thisMapHasNoValue {
					newValue := reflect.MakeMap(currFieldValue.Type())
					currFieldValue.Set(newValue)

					if !steps.moreStepsRemaining() && unset {
						return nil
					}
				}

				if !steps.moreStepsRemaining() && unset {
					// if we're supposed to unset the value or if the value is a map that doesn't exist, create a new value and overwrite
					newValue := reflect.New(currFieldValue.Type()).Elem()
					currFieldValue.Set(newValue)
					return nil
				}

				return modifyConfig(currFieldValue.Addr(), steps, propertyValue, unset, setRawBytes)
			}
		}

		return fmt.Errorf("unable to locate path %#v under %v", currStep, actualCurrValue)

	}

	panic(fmt.Errorf("unrecognized type: %v", actualCurrValue))
}

// getStructFieldIndexByName returns the index number of a field with a name
func getStructFieldIndexByName(objValue reflect.Value, name string) int {
	// Iterate through fields until we find the field we're looking for and return the index
	for fieldIndex := 0; fieldIndex < objValue.NumField(); fieldIndex++ {
		currFieldType := objValue.Type().Field(fieldIndex)
		currYamlTag := currFieldType.Tag.Get("json")
		currFieldTypeYamlName := strings.Split(currYamlTag, ",")[0]
		if currFieldTypeYamlName == name {
			return fieldIndex
		}
	}

	// If we never find the Name key return false
	return -1
}

func editStringSlice(slice []string, input string, deduplicate bool) []string {
	function := string(input[len(input)-1])
	switch function {
	case "-":
		// Remove an argument
		// Remove last character defining function type
		input = input[:len(input)-1]
		argSlice := strings.Split(input, ",")

		for _, arg := range argSlice {
			for j, existingArgs := range slice {
				if existingArgs == arg {
					slice = append(slice[:j], slice[j+1:]...)
					break
				}
			}
		}

		if deduplicate {
			return sets.NewString(slice...).List()
		}
		return slice

	case "+":
		// Add new argument
		// Remove last character defining function type
		input = input[:len(input)-1]
		if len(input) > 0 {
			argSlice := strings.Split(input, ",")
			slice = append(slice, argSlice...)
		}
		if deduplicate {
			return sets.NewString(slice...).List()
		}
		return slice

	default:
		argSlice := strings.Split(input, ",")
		if deduplicate {
			return sets.NewString(argSlice...).List()
		}
		return argSlice
	}
}
