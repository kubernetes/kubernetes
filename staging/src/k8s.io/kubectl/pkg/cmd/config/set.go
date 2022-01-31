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
	"sort"
	"strings"

	"github.com/spf13/cobra"

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
}

var (
	setLong = templates.LongDesc(i18n.T(`
	Set an individual value in a kubeconfig file.

	PROPERTY_NAME is a dot delimited name where each token represents either an attribute name or a map key.  Map keys may not contain dots.

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
	kubectl config set users.cluster-admin.client-key-data cert_data_here --set-raw-bytes=true`)
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

	err = modifyConfig(reflect.ValueOf(config), steps, o.propertyValue, false, setRawBytes)
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

func modifyConfig(curr reflect.Value, steps *navigationSteps, propertyValue string, unset bool, setRawBytes bool) error {
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
					newSliceValue := editStringSlice(currentSliceValue, propertyValue)
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
		if setRawBytes {
			actualCurrValue.SetBytes([]byte(propertyValue))
		} else {
			innerType := actualCurrValue.Type().Elem()
			if innerType.Kind() == reflect.String {
				currentSliceValue := actualCurrValue.Interface().([]string)
				newSliceValue := editStringSlice(currentSliceValue, propertyValue)
				actualCurrValue.Set(reflect.ValueOf(newSliceValue))
			} else if innerType.Kind() == reflect.Struct {
				// Note this only works for attempting to set struct fields of type string
				// Set struct field we are searching on and value we will be searching for
				stepSearchValue := steps.pop()
				searchField := currStep.stepValue
				searchValue := stepSearchValue.stepValue

				if reflect.ValueOf(searchField).Kind() != reflect.String {
					return fmt.Errorf("can not search for fields with non string values")
				}

				// Set struct field and value we will be setting
				if len(strings.Split(propertyValue, ":")) != 2 {
					return fmt.Errorf("error parsing field name for value, should be of format fieldName:fieldValue")
				}
				setField := strings.Split(propertyValue, ":")[0]
				setValue := strings.Split(propertyValue, ":")[1]

				targetStructIndex := getStructByFieldName(actualCurrValue, searchField, searchValue)

				if targetStructIndex < 0 {
					// Set new inner struct value, then outer slice value, then set
					// new struct into new slice and pass to actual curr value
					newSliceValue := reflect.MakeSlice(actualCurrValue.Type(), 0, 0)
					actualCurrValue.Set(reflect.Indirect(newSliceValue))
					targetStructIndex = 0
					newValue := reflect.New(innerType)
					actualCurrValue.Set(reflect.Append(actualCurrValue, reflect.Indirect(newValue)))
					if !steps.moreStepsRemaining() && unset {
						return nil
					}
				}

				targetStruct := actualCurrValue.Index(targetStructIndex)
				searchFieldIndex := getStructFieldIndexByName(targetStruct, searchField)
				if searchFieldIndex < 0 {
					return fmt.Errorf("could not find field in struct with name %v", searchField)
				}
				actualCurrValue.Index(targetStructIndex).FieldByIndex([]int{searchFieldIndex}).Set(reflect.ValueOf(searchValue))

				setFieldIndex := getStructFieldIndexByName(targetStruct, setField)
				if setFieldIndex < 0 {
					return fmt.Errorf("could not find field in struct with name %v", setField)
				}
				actualCurrValue.Index(targetStructIndex).FieldByIndex([]int{setFieldIndex}).Set(reflect.ValueOf(setValue))

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

				return modifyConfig(currFieldValue.Addr(), steps, propertyValue, unset, setRawBytes)
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
		steps.currentStepIndex = steps.currentStepIndex - 1
		return modifyConfig(newActualCurrValue.Addr(), steps, propertyValue, unset, setRawBytes)

	}

	return fmt.Errorf("unrecognized type: %v\nwanted: %v", actualCurrValue, actualCurrValue.Kind())
}

// getStructByFieldName gets the index of the struct in a slice that has the given field name set to the given value
func getStructByFieldName(v reflect.Value, name, value string) int {
	if v.Kind() != reflect.Slice {
		return -1
	}

	// Pull slice value out of value object
	slice, ok := v.Interface().([]interface{})
	if !ok {
		return -1
	}

	// Iterate through slice of ExecEnvVars and check for a matching Name key, return when found
	for i, obj := range slice {
		objValue := reflect.ValueOf(obj)
		for fieldIndex := 0; fieldIndex < objValue.NumField(); fieldIndex++ {
			currFieldValue := objValue.Field(fieldIndex)
			currFieldTypeYamlName := getMapFieldTypeYamlName(objValue, fieldIndex)

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

func dedupeStringSlice(slice []string) []string {
	sliceMap := make(map[string]struct{})
	for i := 0; i < len(slice); i++ {
		sliceMap[slice[i]] = struct{}{}
	}
	var dedupeSlice []string
	for k := range sliceMap {
		dedupeSlice = append(dedupeSlice, k)
	}
	sort.Strings(dedupeSlice)
	return dedupeSlice
}

func editStringSlice(slice []string, input string) []string {
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

		return slice

	case "+":
		// Add new argument
		// Remove last character defining function type
		input = string(input[:len(input)-1])
		argSlice := strings.Split(input, ",")
		slice = append(slice, argSlice...)
		return dedupeStringSlice(slice)

	default:
		argSlice := strings.Split(input, ",")
		return dedupeStringSlice(argSlice)
	}
}
