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

	"k8s.io/apiserver/pkg/util/flag"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

const (
	cannotHaveStepsAfterError                = "Cannot have steps after %v.  %v are remaining"
	additionStepRequiredUnlessUnsettingError = "Must have additional steps after %v unless you are unsetting it"
)

type setOptions struct {
	configAccess  clientcmd.ConfigAccess
	propertyName  string
	propertyValue string
	setRawBytes   flag.Tristate
}

var set_long = templates.LongDesc(`
	Sets an individual value in a kubeconfig file

	PROPERTY_NAME is a dot delimited name where each token represents either an attribute name or a map key.  Map keys may not contain dots.

	PROPERTY_VALUE is the new value you wish to set. Binary fields such as 'certificate-authority-data' expect a base64 encoded string unless the --set-raw-bytes flag is used.`)

func NewCmdConfigSet(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &setOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:   "set PROPERTY_NAME PROPERTY_VALUE",
		Short: i18n.T("Sets an individual value in a kubeconfig file"),
		Long:  set_long,
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
		cmd.Help()
		return fmt.Errorf("Unexpected args: %v", endingArgs)
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
