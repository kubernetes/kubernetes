package hil

import (
	"fmt"
	"reflect"

	"github.com/hashicorp/hil/ast"
	"github.com/mitchellh/mapstructure"
)

var hilMapstructureDecodeHookSlice []interface{}
var hilMapstructureDecodeHookStringSlice []string
var hilMapstructureDecodeHookMap map[string]interface{}

// hilMapstructureWeakDecode behaves in the same way as mapstructure.WeakDecode
// but has a DecodeHook which defeats the backward compatibility mode of mapstructure
// which WeakDecodes []interface{}{} into an empty map[string]interface{}. This
// allows us to use WeakDecode (desirable), but not fail on empty lists.
func hilMapstructureWeakDecode(m interface{}, rawVal interface{}) error {
	config := &mapstructure.DecoderConfig{
		DecodeHook: func(source reflect.Type, target reflect.Type, val interface{}) (interface{}, error) {
			sliceType := reflect.TypeOf(hilMapstructureDecodeHookSlice)
			stringSliceType := reflect.TypeOf(hilMapstructureDecodeHookStringSlice)
			mapType := reflect.TypeOf(hilMapstructureDecodeHookMap)

			if (source == sliceType || source == stringSliceType) && target == mapType {
				return nil, fmt.Errorf("Cannot convert %s into a %s", source, target)
			}

			return val, nil
		},
		WeaklyTypedInput: true,
		Result:           rawVal,
	}

	decoder, err := mapstructure.NewDecoder(config)
	if err != nil {
		return err
	}

	return decoder.Decode(m)
}

func InterfaceToVariable(input interface{}) (ast.Variable, error) {
	if inputVariable, ok := input.(ast.Variable); ok {
		return inputVariable, nil
	}

	var stringVal string
	if err := hilMapstructureWeakDecode(input, &stringVal); err == nil {
		return ast.Variable{
			Type:  ast.TypeString,
			Value: stringVal,
		}, nil
	}

	var mapVal map[string]interface{}
	if err := hilMapstructureWeakDecode(input, &mapVal); err == nil {
		elements := make(map[string]ast.Variable)
		for i, element := range mapVal {
			varElement, err := InterfaceToVariable(element)
			if err != nil {
				return ast.Variable{}, err
			}
			elements[i] = varElement
		}

		return ast.Variable{
			Type:  ast.TypeMap,
			Value: elements,
		}, nil
	}

	var sliceVal []interface{}
	if err := hilMapstructureWeakDecode(input, &sliceVal); err == nil {
		elements := make([]ast.Variable, len(sliceVal))
		for i, element := range sliceVal {
			varElement, err := InterfaceToVariable(element)
			if err != nil {
				return ast.Variable{}, err
			}
			elements[i] = varElement
		}

		return ast.Variable{
			Type:  ast.TypeList,
			Value: elements,
		}, nil
	}

	return ast.Variable{}, fmt.Errorf("value for conversion must be a string, interface{} or map[string]interface: got %T", input)
}

func VariableToInterface(input ast.Variable) (interface{}, error) {
	if input.Type == ast.TypeString {
		if inputStr, ok := input.Value.(string); ok {
			return inputStr, nil
		} else {
			return nil, fmt.Errorf("ast.Variable with type string has value which is not a string")
		}
	}

	if input.Type == ast.TypeList {
		inputList, ok := input.Value.([]ast.Variable)
		if !ok {
			return nil, fmt.Errorf("ast.Variable with type list has value which is not a []ast.Variable")
		}

		result := make([]interface{}, 0)
		if len(inputList) == 0 {
			return result, nil
		}

		for _, element := range inputList {
			if convertedElement, err := VariableToInterface(element); err == nil {
				result = append(result, convertedElement)
			} else {
				return nil, err
			}
		}

		return result, nil
	}

	if input.Type == ast.TypeMap {
		inputMap, ok := input.Value.(map[string]ast.Variable)
		if !ok {
			return nil, fmt.Errorf("ast.Variable with type map has value which is not a map[string]ast.Variable")
		}

		result := make(map[string]interface{}, 0)
		if len(inputMap) == 0 {
			return result, nil
		}

		for key, value := range inputMap {
			if convertedValue, err := VariableToInterface(value); err == nil {
				result[key] = convertedValue
			} else {
				return nil, err
			}
		}

		return result, nil
	}

	return nil, fmt.Errorf("unknown input type: %s", input.Type)
}
