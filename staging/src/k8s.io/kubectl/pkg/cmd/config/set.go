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
	deduplicate   bool
}

var (
	setLong = templates.LongDesc(i18n.T(`
	Set an individual value in a kubeconfig file.

	PROPERTY_NAME is a dot delimited name where each token represents either an attribute name or a map key.  Map keys may not contain dots.

	PROPERTY_VALUE is the new value you want to set. Binary fields such as 'certificate-authority-data' expect a base64 encoded string unless the --set-raw-bytes flag is used.

	When configuring lists you can insert new or remove existing keys by appending a + or - to the comma separated list of values respectively, e.g. value1,value2+ or value1,value2-

	When configuring a list of structs, you will need to specify a structs field and value pair to search for in the dot notation in form of field.value. The value will need to be set by specifying the field you are editing and the value to set it to, separated by a :. e.g. kubectl config set users.foo.exec.env.name.test value:val1

	To delete an existing struct in a list of structs you will leave the value field blank, e.g. kubectl config set users.foo.exec.env.name.test value:

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
	
	# Set a new act-as-group for an existing list of groups
	kubectl config set users.foo.act-as-groups newGroup+
	
	# Remove existing group from list of groups under act-as-group
	kubectl config set users.foo.act-as-groups existingGroup-
	
	# Set an exec environment variable with the name test and the value val1
	kubectl config set users.foo.exec.env.name.test value:val1

	# Delete an exec environment variable with the name test
	kubectl config set users.foo.exec.env.name.test value:

	# Set an exec argument list where the first entry uses a - or --
	kubectl config set users.foo.exec.args -- "--test-arg,next,thing,-i"`)
)

// NewCmdConfigSet returns a Command instance for 'config set' sub command
func NewCmdConfigSet(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &setOptions{configAccess: configAccess}

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
	cmd.Flags().BoolVar(&options.deduplicate, "deduplicate", false, "Whether to deduplicate and sort the argument list or not")

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
	steps, err := newNavigationSteps(o.propertyName)
	if err != nil {
		return err
	}

	setRawBytes := false
	if o.setRawBytes.Provided() {
		setRawBytes = o.setRawBytes.Value()
	}

	err = modifyConfig(reflect.ValueOf(config), steps, o.propertyValue, false, setRawBytes, o.deduplicate)
	if err != nil {
		return err
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

func modifyConfig(curr reflect.Value, steps *navigationSteps, propertyValue string, unset bool, setRawBytes bool, deduplicate bool) error {
	currStep := steps.pop()

	actualCurrValue := curr
	if curr.Kind() == reflect.Ptr {
		actualCurrValue = curr.Elem()
	}

	switch actualCurrValue.Kind() {
	case reflect.Map:
		mapKey := reflect.ValueOf(currStep.stepValue)
		mapValueType := curr.Type().Elem().Elem()
		currMapValue := actualCurrValue.MapIndex(mapKey)

		if !steps.moreStepsRemaining() && !unset {
			switch mapValueType.Kind() {
			case reflect.String:
				if currStep.stepValue == "" {
					return fmt.Errorf("empty key provided for map")
				}
				actualCurrValue.SetMapIndex(reflect.ValueOf(currStep.stepValue), reflect.ValueOf(propertyValue))

			case reflect.Slice:
				sliceType := mapValueType.Elem()
				switch sliceType.Kind() {
				case reflect.String:
					needToSetNewMapValue := currMapValue.Kind() == reflect.Invalid
					if needToSetNewMapValue {
						if unset {
							return fmt.Errorf("current map key `%v` is invalid", mapKey.Interface())
						}
						currMapValue = reflect.New(mapValueType)
						actualCurrValue.SetMapIndex(mapKey, reflect.Indirect(currMapValue))
					}
					currentSliceValue := actualCurrValue.MapIndex(mapKey).Interface().([]string)
					newSliceValue := editStringSlice(currentSliceValue, propertyValue, deduplicate)
					actualCurrValue.SetMapIndex(mapKey, reflect.ValueOf(newSliceValue))

				default:
					return fmt.Errorf("could not parse type of map value")
				}
			}
			return nil
		}

		if !steps.moreStepsRemaining() && unset {
			actualCurrValue.SetMapIndex(mapKey, reflect.Value{})
			return nil
		}

		needToSetNewMapValue := currMapValue.Kind() == reflect.Invalid
		if needToSetNewMapValue {
			if unset {
				return fmt.Errorf("current map key `%v` is invalid", mapKey.Interface())
			}
			currMapValue = reflect.New(mapValueType.Elem()).Elem().Addr()
			actualCurrValue.SetMapIndex(mapKey, currMapValue)
		}

		err := modifyConfig(currMapValue, steps, propertyValue, unset, setRawBytes, deduplicate)
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
		if setRawBytes {
			actualCurrValue.SetBytes([]byte(propertyValue))
		} else {
			innerType := actualCurrValue.Type().Elem()
			if innerType.Kind() == reflect.String {
				if currStep.stepValue != "" {
					return fmt.Errorf("unable to locate path %v", currStep.stepValue)
				}
				currentSliceValue := actualCurrValue.Interface().([]string)
				newSliceValue := editStringSlice(currentSliceValue, propertyValue, deduplicate)
				actualCurrValue.Set(reflect.ValueOf(newSliceValue))
			} else if innerType.Kind() == reflect.Struct {
				// Note this only works for attempting to set struct fields of type string
				// Set struct field we are searching on and value we will be searching for
				stepSearchValue := steps.pop()
				if steps.moreStepsRemaining() {
					errorStep := steps.pop()
					return fmt.Errorf("unable to locate path %v", errorStep.stepValue)
				}

				searchField := currStep.stepValue
				searchValue := stepSearchValue.stepValue

				if reflect.ValueOf(searchField).Kind() != reflect.String {
					return fmt.Errorf("can not search for fields with non string values")
				}

				// Set struct field and value we will be setting
				valueSlice := strings.Split(propertyValue, ":")
				if len(valueSlice) != 2 {
					return fmt.Errorf("error parsing field name for value, should be of format fieldName:fieldValue")
				}
				setField := valueSlice[0]
				setValue := valueSlice[1]

				if setValue == "" {
					targetStructIndex := getStructByFieldName(actualCurrValue, searchField, searchValue)
					if targetStructIndex < 0 {
						return nil
					}
					newValueFirstHalf := reflect.MakeSlice(actualCurrValue.Type(), 0, 0)
					if targetStructIndex != 0 {
						newValueFirstHalf = actualCurrValue.Slice(0, targetStructIndex)
					}
					newValueSecondHalf := reflect.MakeSlice(actualCurrValue.Type(), 0, 0)
					if targetStructIndex != actualCurrValue.Len() {
						newValueSecondHalf = actualCurrValue.Slice(targetStructIndex+1, actualCurrValue.Len())
					} else {
						newValueSecondHalf = actualCurrValue.Slice(targetStructIndex+1, actualCurrValue.Len()-1)
					}
					actualCurrValue.Set(reflect.AppendSlice(newValueFirstHalf, newValueSecondHalf))
					return nil
				}

				targetStructIndex := 0

				if actualCurrValue.IsZero() {
					// Set new inner struct value, then outer slice value, then set
					// new struct into new slice and pass to actual curr value
					newSliceValue := reflect.MakeSlice(actualCurrValue.Type(), 0, 0)
					actualCurrValue.Set(reflect.Indirect(newSliceValue))
					newValue := reflect.New(innerType)
					actualCurrValue.Set(reflect.Append(actualCurrValue, reflect.Indirect(newValue)))
					if !steps.moreStepsRemaining() && unset {
						return nil
					}

					err := setStructInSlice(actualCurrValue, targetStructIndex, searchField, searchValue, setField, setValue)
					if err != nil {
						return err
					}
					return nil
				}

				targetStructIndex = getStructByFieldName(actualCurrValue, searchField, searchValue)

				if targetStructIndex < 0 {
					targetStructIndex = actualCurrValue.Len()

					// Just set the new inner struct value
					newValue := reflect.Indirect(reflect.New(innerType))
					actualCurrValue.Set(reflect.Append(actualCurrValue, newValue))

					err := setStructInSlice(actualCurrValue, targetStructIndex, searchField, searchValue, setField, setValue)
					if err != nil {
						return err
					}
					return nil
				}

				err := setStructInSlice(actualCurrValue, targetStructIndex, searchField, searchValue, setField, setValue)
				if err != nil {
					return err
				}
				return nil

			} else {
				val, err := base64.StdEncoding.DecodeString(propertyValue)
				if err != nil {
					return fmt.Errorf("error decoding input value: %v", err)
				}
				actualCurrValue.SetBytes(val)
			}
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
			currFieldTypeYamlName := getMapFieldTypeYamlName(actualCurrValue, fieldIndex)

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

				return modifyConfig(currFieldValue.Addr(), steps, propertyValue, unset, setRawBytes, deduplicate)
			}
		}

		return fmt.Errorf("unable to locate path %v", currStep.stepValue)

	case reflect.Ptr:
		newActualCurrValue := actualCurrValue.Elem()
		if actualCurrValue.IsNil() {
			newValue := reflect.New(actualCurrValue.Type().Elem())
			actualCurrValue.Set(newValue)
			newActualCurrValue = actualCurrValue.Elem()
			if !steps.moreStepsRemaining() && unset {
				return nil
			}
		}
		steps.currentStepIndex -= 1
		return modifyConfig(newActualCurrValue.Addr(), steps, propertyValue, unset, setRawBytes, deduplicate)

	}

	return fmt.Errorf("unrecognized type: %v\nwanted: %v", actualCurrValue, actualCurrValue.Kind())
}

// getStructByFieldName gets the index of the struct in a slice that has the given field name set to the given value
func getStructByFieldName(v reflect.Value, name, value string) int {
	if v.Kind() != reflect.Slice {
		return -1
	}

	// Iterate through slice of structs and check for a matching Name key, return when found
	sliceLen := v.Len()
	for i := 0; i < sliceLen; i++ {
		searchStructValue := v.Index(i)
		structPropFieldIndex := getStructFieldIndexByName(searchStructValue, name)
		if structPropFieldIndex >= 0 {
			currFieldValue := searchStructValue.Field(structPropFieldIndex)
			currFieldTypeYamlName := getMapFieldTypeYamlName(searchStructValue, structPropFieldIndex)

			if currFieldTypeYamlName == name && currFieldValue.String() == value {
				return i
			}
		}
	}

	// If we never find the Name key return false
	return -1
}

// getStructFieldIndexByName returns the index number of a field with a name
func getStructFieldIndexByName(objValue reflect.Value, name string) int {
	// Iterate through fields until we find the field we're looking for and return the index
	for fieldIndex := 0; fieldIndex < objValue.NumField(); fieldIndex++ {
		currFieldTypeYamlName := getMapFieldTypeYamlName(objValue, fieldIndex)
		if currFieldTypeYamlName == name {
			return fieldIndex
		}
	}

	// If we never find the Name key return false
	return -1
}

func getMapFieldTypeYamlName(objValue reflect.Value, fieldIndex int) string {
	currFieldType := objValue.Type().Field(fieldIndex)
	currYamlTag := currFieldType.Tag.Get("json")
	currFieldTypeYamlName := strings.Split(currYamlTag, ",")[0]
	return currFieldTypeYamlName
}

func editStringSlice(slice []string, input string, deduplicate bool) []string {
	function := string(input[len(input)-1])
	switch function {
	case "-":
		// Remove an argument
		// Remove last character defining function type
		input = string(input[:len(input)-1])
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
		input = string(input[:len(input)-1])
		argSlice := strings.Split(input, ",")
		slice = append(slice, argSlice...)
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

func setStructInSlice(currSliceValue reflect.Value, structIndex int, searchField, searchValue, setField, setValue string) error {
	targetStruct := currSliceValue.Index(structIndex)
	searchFieldIndex := getStructFieldIndexByName(targetStruct, searchField)
	if searchFieldIndex < 0 {
		return fmt.Errorf("could not find field in struct with name %v", searchField)
	}
	currSliceValue.Index(structIndex).FieldByIndex([]int{searchFieldIndex}).Set(reflect.ValueOf(searchValue))

	setFieldIndex := getStructFieldIndexByName(targetStruct, setField)
	if setFieldIndex < 0 {
		return fmt.Errorf("could not find field in struct with name %v", setField)
	}
	currSliceValue.Index(structIndex).FieldByIndex([]int{setFieldIndex}).Set(reflect.ValueOf(setValue))

	return nil
}
