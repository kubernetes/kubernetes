package flags

import (
	"encoding/json"
	"reflect"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/pflag"
)

// OutputFlags contains information for specifying multiple test names.
type OutputFlags struct {
	Output string
}

func NewOutputFlags() *OutputFlags {
	return &OutputFlags{
		Output: "json",
	}
}

func (f *OutputFlags) BindFlags(fs *pflag.FlagSet) {
	fs.StringVarP(&f.Output,
		"output",
		"o",
		f.Output,
		"output mode")
}

func (o *OutputFlags) Marshal(v interface{}) ([]byte, error) {
	switch o.Output {
	case "", "json":
		j, err := json.MarshalIndent(&v, "", "  ")
		if err != nil {
			return nil, err
		}
		return j, nil
	case "jsonl":
		// Check if v is a slice or array
		val := reflect.ValueOf(v)
		if val.Kind() == reflect.Slice || val.Kind() == reflect.Array {
			var result []byte
			for i := 0; i < val.Len(); i++ {
				item := val.Index(i).Interface()
				j, err := json.Marshal(item)
				if err != nil {
					return nil, err
				}
				result = append(result, j...)
				result = append(result, '\n') // Append newline after each item
			}
			return result, nil
		}
		return nil, errors.New("jsonl format requires a slice or array")
	case "names":
		val := reflect.ValueOf(v)
		if val.Kind() == reflect.Slice || val.Kind() == reflect.Array {
			var names []string
		outerLoop:
			for i := 0; i < val.Len(); i++ {
				item := val.Index(i)
				// Check for Name() or Identifier() methods
				itemInterface := item.Interface()
				nameFuncs := []string{"Name", "Identifier"}
				for _, fn := range nameFuncs {
					method := reflect.ValueOf(itemInterface).MethodByName(fn)
					if method.IsValid() && method.Kind() == reflect.Func && method.Type().NumIn() == 0 && method.Type().NumOut() == 1 && method.Type().Out(0).Kind() == reflect.String {
						name := method.Call(nil)[0].String()
						names = append(names, name)
						continue outerLoop
					}
				}

				// Dereference pointer if needed
				if item.Kind() == reflect.Ptr {
					item = item.Elem()
				}
				// Check for struct with Name field
				if item.Kind() == reflect.Struct {
					nameField := item.FieldByName("Name")
					if nameField.IsValid() && nameField.Kind() == reflect.String {
						names = append(names, nameField.String())
					}
				} else {
					return nil, errors.New("items must have a Name field or a Name() method")
				}
			}
			return []byte(strings.Join(names, "\n")), nil
		}
		return nil, errors.New("names format requires an array of structs")
	default:
		return nil, errors.Errorf("invalid output format: %s", o.Output)
	}
}
