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

package config

import (
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
)

const (
	cannotHaveStepsAfterError                = "Cannot have steps after %v.  %v are remaining"
	additionStepRequiredUnlessUnsettingError = "Must have additional steps after %v unless you are unsetting it"
)

type navigationSteps []string

type setOptions struct {
	pathOptions   *pathOptions
	propertyName  string
	propertyValue string
}

func NewCmdConfigSet(out io.Writer, pathOptions *pathOptions) *cobra.Command {
	options := &setOptions{pathOptions: pathOptions}

	cmd := &cobra.Command{
		Use:   "set property-name property-value",
		Short: "Sets an individual value in a .kubeconfig file",
		Long: `Sets an individual value in a .kubeconfig file

		property-name is a dot delimitted name where each token represents either a attribute name or a map key.  Map keys may not contain dots.
		property-value is the new value you wish to set.

		`,
		Run: func(cmd *cobra.Command, args []string) {
			if !options.complete(cmd) {
				return
			}

			err := options.run()
			if err != nil {
				fmt.Printf("%v\n", err)
			}
		},
	}

	return cmd
}

func (o setOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, filename, err := o.pathOptions.getStartingConfig()
	if err != nil {
		return err
	}

	if len(filename) == 0 {
		return errors.New("cannot set property without using a specific file")
	}

	parts := strings.Split(o.propertyName, ".")
	err = modifyConfig(reflect.ValueOf(config), parts, o.propertyValue, false)
	if err != nil {
		return err
	}

	err = clientcmd.WriteToFile(*config, filename)
	if err != nil {
		return err
	}

	return nil
}

func (o *setOptions) complete(cmd *cobra.Command) bool {
	endingArgs := cmd.Flags().Args()
	if len(endingArgs) != 2 {
		cmd.Help()
		return false
	}

	o.propertyValue = endingArgs[1]
	o.propertyName = endingArgs[0]
	return true
}

func (o setOptions) validate() error {
	if len(o.propertyValue) == 0 {
		return errors.New("You cannot use set to unset a property")
	}

	if len(o.propertyName) == 0 {
		return errors.New("You must specify a property")
	}

	return nil
}

// moreStepsRemaining just makes code read cleaner
func moreStepsRemaining(remainder []string) bool {
	return len(remainder) != 0
}

func (s navigationSteps) nextSteps() navigationSteps {
	if len(s) < 2 {
		return make([]string, 0, 0)
	} else {
		return s[1:]
	}
}
func (s navigationSteps) moreStepsRemaining() bool {
	return len(s) != 0
}
func (s navigationSteps) nextStep() string {
	return s[0]
}

func modifyConfig(curr reflect.Value, steps navigationSteps, propertyValue string, unset bool) error {
	shouldUnsetNextField := !steps.nextSteps().moreStepsRemaining() && unset
	shouldSetThisField := !steps.moreStepsRemaining() && !unset

	actualCurrValue := curr
	if curr.Kind() == reflect.Ptr {
		actualCurrValue = curr.Elem()
	}

	switch actualCurrValue.Kind() {
	case reflect.Map:
		if shouldSetThisField {
			return fmt.Errorf("Can't set a map to a value: %v", actualCurrValue)
		}

		mapKey := reflect.ValueOf(steps.nextStep())
		mapValueType := curr.Type().Elem().Elem()

		if shouldUnsetNextField {
			actualCurrValue.SetMapIndex(mapKey, reflect.Value{})
			return nil
		}

		currMapValue := actualCurrValue.MapIndex(mapKey)

		needToSetNewMapValue := currMapValue.Kind() == reflect.Invalid
		if needToSetNewMapValue {
			currMapValue = reflect.New(mapValueType).Elem()
			actualCurrValue.SetMapIndex(mapKey, currMapValue)
		}

		// our maps do not hold pointers to structs, they hold the structs themselves.  This means that MapIndex returns the struct itself
		// That in turn means that they have kinds of type.Struct, which is not a settable type.  Because of this, we need to make new struct of that type
		// copy all the data from the old value into the new value, then take the .addr of the new value to modify it in the next recursion.
		// clear as mud
		modifiableMapValue := reflect.New(currMapValue.Type()).Elem()
		modifiableMapValue.Set(currMapValue)

		if modifiableMapValue.Kind() == reflect.Struct {
			modifiableMapValue = modifiableMapValue.Addr()
		}
		err := modifyConfig(modifiableMapValue, steps.nextSteps(), propertyValue, unset)
		if err != nil {
			return err
		}

		actualCurrValue.SetMapIndex(mapKey, reflect.Indirect(modifiableMapValue))
		return nil

	case reflect.String:
		if steps.moreStepsRemaining() {
			return fmt.Errorf("Can't have more steps after a string. %v", steps)
		}
		actualCurrValue.SetString(propertyValue)
		return nil

	case reflect.Bool:
		if steps.moreStepsRemaining() {
			return fmt.Errorf("Can't have more steps after a bool. %v", steps)
		}
		boolValue, err := toBool(propertyValue)
		if err != nil {
			return err
		}
		actualCurrValue.SetBool(boolValue)
		return nil

	case reflect.Struct:
		if !steps.moreStepsRemaining() {
			return fmt.Errorf("Can't set a struct to a value: %v", actualCurrValue)
		}

		for fieldIndex := 0; fieldIndex < actualCurrValue.NumField(); fieldIndex++ {
			currFieldValue := actualCurrValue.Field(fieldIndex)
			currFieldType := actualCurrValue.Type().Field(fieldIndex)
			currYamlTag := currFieldType.Tag.Get("json")
			currFieldTypeYamlName := strings.Split(currYamlTag, ",")[0]

			if currFieldTypeYamlName == steps.nextStep() {
				thisMapHasNoValue := (currFieldValue.Kind() == reflect.Map && currFieldValue.IsNil())

				if thisMapHasNoValue {
					newValue := reflect.MakeMap(currFieldValue.Type())
					currFieldValue.Set(newValue)

					if shouldUnsetNextField {
						return nil
					}
				}

				if shouldUnsetNextField {
					// if we're supposed to unset the value or if the value is a map that doesn't exist, create a new value and overwrite
					newValue := reflect.New(currFieldValue.Type()).Elem()
					currFieldValue.Set(newValue)
					return nil
				}

				return modifyConfig(currFieldValue.Addr(), steps.nextSteps(), propertyValue, unset)
			}
		}

		return fmt.Errorf("Unable to locate path %v under %v", steps, actualCurrValue)

	}

	return fmt.Errorf("Unrecognized type: %v", actualCurrValue)
}
