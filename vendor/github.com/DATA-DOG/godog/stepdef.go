package godog

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/DATA-DOG/godog/gherkin"
)

var matchFuncDefRef = regexp.MustCompile(`\(([^\)]+)\)`)

// Steps allows to nest steps
// instead of returning an error in step func
// it is possible to return combined steps:
//
//   func multistep(name string) godog.Steps {
//     return godog.Steps{
//       fmt.Sprintf(`an user named "%s"`, name),
//       fmt.Sprintf(`user "%s" is authenticated`, name),
//     }
//   }
//
// These steps will be matched and executed in
// sequential order. The first one which fails
// will result in main step failure.
type Steps []string

// StepDef is a registered step definition
// contains a StepHandler and regexp which
// is used to match a step. Args which
// were matched by last executed step
//
// This structure is passed to the formatter
// when step is matched and is either failed
// or successful
type StepDef struct {
	args    []interface{}
	hv      reflect.Value
	Expr    *regexp.Regexp
	Handler interface{}

	// multistep related
	nested    bool
	undefined []string
}

func (sd *StepDef) definitionID() string {
	ptr := sd.hv.Pointer()
	f := runtime.FuncForPC(ptr)
	file, line := f.FileLine(ptr)
	dir := filepath.Dir(file)

	fn := strings.Replace(f.Name(), dir, "", -1)
	var parts []string
	for _, gr := range matchFuncDefRef.FindAllStringSubmatch(fn, -1) {
		parts = append(parts, strings.Trim(gr[1], "_."))
	}
	if len(parts) > 0 {
		// case when suite is a structure with methods
		fn = strings.Join(parts, ".")
	} else {
		// case when steps are just plain funcs
		fn = strings.Trim(fn, "_.")
	}

	if pkg := os.Getenv("GODOG_TESTED_PACKAGE"); len(pkg) > 0 {
		fn = strings.Replace(fn, pkg, "", 1)
		fn = strings.TrimLeft(fn, ".")
		fn = strings.Replace(fn, "..", ".", -1)
	}

	return fmt.Sprintf("%s:%d -> %s", filepath.Base(file), line, fn)
}

// run a step with the matched arguments using
// reflect
func (sd *StepDef) run() interface{} {
	typ := sd.hv.Type()
	if len(sd.args) < typ.NumIn() {
		return fmt.Errorf("func expects %d arguments, which is more than %d matched from step", typ.NumIn(), len(sd.args))
	}
	var values []reflect.Value
	for i := 0; i < typ.NumIn(); i++ {
		param := typ.In(i)
		switch param.Kind() {
		case reflect.Int:
			s, err := sd.shouldBeString(i)
			if err != nil {
				return err
			}
			v, err := strconv.ParseInt(s, 10, 0)
			if err != nil {
				return fmt.Errorf(`cannot convert argument %d: "%s" to int: %s`, i, s, err)
			}
			values = append(values, reflect.ValueOf(int(v)))
		case reflect.Int64:
			s, err := sd.shouldBeString(i)
			if err != nil {
				return err
			}
			v, err := strconv.ParseInt(s, 10, 64)
			if err != nil {
				return fmt.Errorf(`cannot convert argument %d: "%s" to int64: %s`, i, s, err)
			}
			values = append(values, reflect.ValueOf(int64(v)))
		case reflect.Int32:
			s, err := sd.shouldBeString(i)
			if err != nil {
				return err
			}
			v, err := strconv.ParseInt(s, 10, 32)
			if err != nil {
				return fmt.Errorf(`cannot convert argument %d: "%s" to int32: %s`, i, s, err)
			}
			values = append(values, reflect.ValueOf(int32(v)))
		case reflect.Int16:
			s, err := sd.shouldBeString(i)
			if err != nil {
				return err
			}
			v, err := strconv.ParseInt(s, 10, 16)
			if err != nil {
				return fmt.Errorf(`cannot convert argument %d: "%s" to int16: %s`, i, s, err)
			}
			values = append(values, reflect.ValueOf(int16(v)))
		case reflect.Int8:
			s, err := sd.shouldBeString(i)
			if err != nil {
				return err
			}
			v, err := strconv.ParseInt(s, 10, 8)
			if err != nil {
				return fmt.Errorf(`cannot convert argument %d: "%s" to int8: %s`, i, s, err)
			}
			values = append(values, reflect.ValueOf(int8(v)))
		case reflect.String:
			s, err := sd.shouldBeString(i)
			if err != nil {
				return err
			}
			values = append(values, reflect.ValueOf(s))
		case reflect.Float64:
			s, err := sd.shouldBeString(i)
			if err != nil {
				return err
			}
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return fmt.Errorf(`cannot convert argument %d: "%s" to float64: %s`, i, s, err)
			}
			values = append(values, reflect.ValueOf(v))
		case reflect.Float32:
			s, err := sd.shouldBeString(i)
			if err != nil {
				return err
			}
			v, err := strconv.ParseFloat(s, 32)
			if err != nil {
				return fmt.Errorf(`cannot convert argument %d: "%s" to float32: %s`, i, s, err)
			}
			values = append(values, reflect.ValueOf(float32(v)))
		case reflect.Ptr:
			arg := sd.args[i]
			switch param.Elem().String() {
			case "gherkin.DocString":
				v, ok := arg.(*gherkin.DocString)
				if !ok {
					return fmt.Errorf(`cannot convert argument %d: "%v" of type "%T" to *gherkin.DocString`, i, arg, arg)
				}
				values = append(values, reflect.ValueOf(v))
			case "gherkin.DataTable":
				v, ok := arg.(*gherkin.DataTable)
				if !ok {
					return fmt.Errorf(`cannot convert argument %d: "%v" of type "%T" to *gherkin.DocString`, i, arg, arg)
				}
				values = append(values, reflect.ValueOf(v))
			default:
				return fmt.Errorf("the argument %d type %T is not supported", i, arg)
			}
		case reflect.Slice:
			switch param {
			case typeOfBytes:
				s, err := sd.shouldBeString(i)
				if err != nil {
					return err
				}
				values = append(values, reflect.ValueOf([]byte(s)))
			default:
				return fmt.Errorf("the slice argument %d type %s is not supported", i, param.Kind())
			}
		default:
			return fmt.Errorf("the argument %d type %s is not supported", i, param.Kind())
		}
	}
	return sd.hv.Call(values)[0].Interface()
}

func (sd *StepDef) shouldBeString(idx int) (string, error) {
	arg := sd.args[idx]
	s, ok := arg.(string)
	if !ok {
		return "", fmt.Errorf(`cannot convert argument %d: "%v" of type "%T" to string`, idx, arg, arg)
	}
	return s, nil
}
