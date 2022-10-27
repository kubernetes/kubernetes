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
	"fmt"
	"reflect"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	clientcmdapiv1 "k8s.io/client-go/tools/clientcmd/api/v1"
	"k8s.io/client-go/util/jsonpath"
)

// modifyConfigJson: This functions enables setting and unsetting values in a given
// k8s.io/client-go/tools/clientcmd/api/v1 config object.
//
// config: the v1 Config object to work on
// propertyName: the jsonpath to use to find and set values
// propertyValue: the value you wish to set in the value found by propertyName
// unset: whether we are attempting to set or unset the value
// setRawBytes: whether we are setting []byte properties with or without base64 conversion
// deduplicate: whether we and to sort and deduplicate the values or not
//
// The intention of this function is to allow uesrs to add new or existing known properties to a kubeconfig file. It is
// expected that a user will pass a jsonpath string that will be to set properties that may not exist yet, and we need
// to support their ability to set these properties, and build the properties along the path that need to exist for the
// desired property to be set. Additionally, users will expect that they will be able to pass multiple top level
// jsonpath nodes, which will look like '{.key1}{.key2}...{.keyn}'. This is why jsonpath results are returned using
// [][]reflect.Value. The first slice divides the string into individual path's, e.g. position 0 of the example is
// {.key1} and so on. The inner slice will contain a list of the results of that the path within the {}, which can be
// one or more, e.g. if we were to use a wildcard like {.key1[*]}, this will return all values in a list, all of which
// the user will expect to be acted on. To support both of these features we must use several nested loops
//
// for:
//   The first level of the for loop supports traversing all individual path's provided, e.g. each of
//   {.key1}{.key2}...{.keyn}
//
//   for:
//     The second level of the for loop supports traversing the actual paths within the {}, e.g. '.path.to.key.to.set'
//     that will get broken up into path, to, key, to, and set.
//
//     for:
//       The third level of the for loop supports traversing the results of the current step in the jsonpath we are at.
//       This is only required because we need to support setting properties in the event a filter node returns nothing
//       which results in the outer results slice having a length of zero.
//
//       for:
//         The fourth and final level of the for loops supports traversing the actual results which will have the values
//         that can actually be set. This is where the meat of the logic is.
//
// Inside the final for loop is a large switch block to handle different types of potential values. These are as
// follows:
//   String - This should always be the value gotten by the last node in the jsonpath
//   Bool - Convert the string input from the user into a bool if we can, then set. this should always be the value
//          gotten by the last node in the jsonpath
//   Pointer - This should always be followed by a field node, and unless we are unsetting should never be the last node
//             in a path. We only need to set anything here if the pointer is zero, so that we get results on the next
//             step in the jsonpath node list
//   Struct - Similar to the Pointer case, this node can not be the final node in the jsonpath unless we are unsetting
//            the value. We do not work on this result and instead only validate that the next node in the path is valid
//   Map - We have two cases to support for this type.
//     * [string]string - This needs to make sure the node that returned the map is the second to last node, with the
//                        last node being the key to set the value to.
//     * [string][]string - This case supports unsetting the entire map if it is the last node in the list, unsetting a
//                          specific key in the map if it is the second to last node in the list, or setting a value to
//                          the map's key
//   Slice - Similar to Map this type has several subtypes.
//     * String - Similar to the top level string type, this must be the final node in the list. This case supports
//                users providing values in a CSV format and will split them and set the strings as desired. This also
//                supports adding and removing values from existing slices by appending + or - to the end of the list
//     * Uint8 - Similar to the string type, this must be the final node in the list. This case is specifically for
//               supporting users setting []byte data and supports setting the data either directly using --set-raw-bytes
//               or will convert the data for the user if the flag is not provided.
//     * Struct - This can not be the final node in the list unless we are unsetting the node. We simply pass on this
//                unless we are unsetting it.
func modifyConfigJson(config *clientcmdapiv1.Config, propertyName, propertyValue string, unset, setRawBytes, deduplicate bool) error {
	// Create a jsonpath parser to use throughout the function. This one will be used for getting values.
	jsonPath := jsonpath.New("Value Getter").AllowMissingKeys(true)

	// Create a jsonpath parser to use throughout the function. This one will be used for parsing nodes from a jsonpath string.
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
			// any other types of nodes and returning an error specifying what type of node was unsupported.
			// If it is required in the future we can add support for other node types after property defining
			// use case and scope.
			var filterKey string
			var filterValue string

			// Checking types so we can build the jsonPathTraverser accurately
			switch node.Type() {
			case jsonpath.NodeField:
				// Add . to field node values because they need to match the form .value
				jsonPathTraverser = append(jsonPathTraverser, "."+node.(*jsonpath.FieldNode).Value)

			case jsonpath.NodeArray:
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

			// Get a list of result values to work on from the current position in the jsonpath
			if err := jsonPath.Parse("{" + strings.Join(jsonPathTraverser, "") + "}"); err != nil {
				return err
			}
			results, err := jsonPath.FindResults(config)
			if err != nil {
				return err
			}

			// We must now work on the results gotten by the current node in the user provided jsonpath
			for _, outerResult := range results {
				// no reason to keep going setting things to unset something eventually so just short circuit
				if len(outerResult) == 0 && unset {
					return nil
				}
				// This should really only apply to the slices of named structs e.g. NamedClusters. It is a result of
				// filter returning an empty list in the results.
				if len(outerResult) == 0 && node.Type() == jsonpath.NodeFilter {
					// Need to back up so that we can create the new named struct type inside the outer list
					// This requires a new jsonpath object so that we can get the new list of results without losing the
					// list of results for the jsonpath node we're currently working on
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
					// This will most likely be setting the "name" field but want to support any filter field.
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
							if nodeIterator < len(innerNodeList)-1 {
								return fmt.Errorf("can't have more nodes after a string slice, %q", strings.Join(jsonPathTraverser[0:nodeIterator], ""))
							}
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
							// This check isn't necessary now as there are no instances of any other map of slices that
							// isn't a map[string][]string, but just "for the future"
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
						// If this is the last node in the jsonpath, and we're trying to unset then unset
						if nodeIterator == len(innerNodeList)-1 && unset {
							innerResult.Set(reflect.Zero(innerResult.Type()))
						}
						if nodeIterator == len(innerNodeList)-1 && !unset {
							return fmt.Errorf("%q cannot be the last node in the path", jsonPathTraverser[nodeIterator])
						}
						// Check to see if next filter value actually exists as a field for the given struct pointer
						if innerNodeList[nodeIterator+1].Type() != jsonpath.NodeField {
							return fmt.Errorf("invalid node type after %q, must be field node", strings.Join(jsonPathTraverser[0:nodeIterator], ""))
						}
						// Check to see if the next node is a valid field in the type pointed to
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
						} else if nodeIterator < len(innerNodeList)-1 && innerNodeList[nodeIterator+1].Type() != jsonpath.NodeField {
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
	if curr.Kind() == reflect.Ptr {
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
