// +build windows

/*
Package wmi provides a WQL interface for WMI on Windows.

Example code to print names of running processes:

	type Win32_Process struct {
		Name string
	}

	func main() {
		var dst []Win32_Process
		q := wmi.CreateQuery(&dst, "")
		err := wmi.Query(q, &dst)
		if err != nil {
			log.Fatal(err)
		}
		for i, v := range dst {
			println(i, v.Name)
		}
	}

*/
package wmi

import (
	"bytes"
	"errors"
	"fmt"
	"log"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-ole/go-ole"
	"github.com/go-ole/go-ole/oleutil"
)

var l = log.New(os.Stdout, "", log.LstdFlags)

var (
	ErrInvalidEntityType = errors.New("wmi: invalid entity type")
	// ErrNilCreateObject is the error returned if CreateObject returns nil even
	// if the error was nil.
	ErrNilCreateObject = errors.New("wmi: create object returned nil")
	lock               sync.Mutex
)

// S_FALSE is returned by CoInitializeEx if it was already called on this thread.
const S_FALSE = 0x00000001

// QueryNamespace invokes Query with the given namespace on the local machine.
func QueryNamespace(query string, dst interface{}, namespace string) error {
	return Query(query, dst, nil, namespace)
}

// Query runs the WQL query and appends the values to dst.
//
// dst must have type *[]S or *[]*S, for some struct type S. Fields selected in
// the query must have the same name in dst. Supported types are all signed and
// unsigned integers, time.Time, string, bool, or a pointer to one of those.
// Array types are not supported.
//
// By default, the local machine and default namespace are used. These can be
// changed using connectServerArgs. See
// http://msdn.microsoft.com/en-us/library/aa393720.aspx for details.
//
// Query is a wrapper around DefaultClient.Query.
func Query(query string, dst interface{}, connectServerArgs ...interface{}) error {
	if DefaultClient.SWbemServicesClient == nil {
		return DefaultClient.Query(query, dst, connectServerArgs...)
	}
	return DefaultClient.SWbemServicesClient.Query(query, dst, connectServerArgs...)
}

// A Client is an WMI query client.
//
// Its zero value (DefaultClient) is a usable client.
type Client struct {
	// NonePtrZero specifies if nil values for fields which aren't pointers
	// should be returned as the field types zero value.
	//
	// Setting this to true allows stucts without pointer fields to be used
	// without the risk failure should a nil value returned from WMI.
	NonePtrZero bool

	// PtrNil specifies if nil values for pointer fields should be returned
	// as nil.
	//
	// Setting this to true will set pointer fields to nil where WMI
	// returned nil, otherwise the types zero value will be returned.
	PtrNil bool

	// AllowMissingFields specifies that struct fields not present in the
	// query result should not result in an error.
	//
	// Setting this to true allows custom queries to be used with full
	// struct definitions instead of having to define multiple structs.
	AllowMissingFields bool

	// SWbemServiceClient is an optional SWbemServices object that can be
	// initialized and then reused across multiple queries. If it is null
	// then the method will initialize a new temporary client each time.
	SWbemServicesClient *SWbemServices
}

// DefaultClient is the default Client and is used by Query, QueryNamespace
var DefaultClient = &Client{}

// Query runs the WQL query and appends the values to dst.
//
// dst must have type *[]S or *[]*S, for some struct type S. Fields selected in
// the query must have the same name in dst. Supported types are all signed and
// unsigned integers, time.Time, string, bool, or a pointer to one of those.
// Array types are not supported.
//
// By default, the local machine and default namespace are used. These can be
// changed using connectServerArgs. See
// http://msdn.microsoft.com/en-us/library/aa393720.aspx for details.
func (c *Client) Query(query string, dst interface{}, connectServerArgs ...interface{}) error {
	dv := reflect.ValueOf(dst)
	if dv.Kind() != reflect.Ptr || dv.IsNil() {
		return ErrInvalidEntityType
	}
	dv = dv.Elem()
	mat, elemType := checkMultiArg(dv)
	if mat == multiArgTypeInvalid {
		return ErrInvalidEntityType
	}

	lock.Lock()
	defer lock.Unlock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	err := ole.CoInitializeEx(0, ole.COINIT_MULTITHREADED)
	if err != nil {
		oleCode := err.(*ole.OleError).Code()
		if oleCode != ole.S_OK && oleCode != S_FALSE {
			return err
		}
	}
	defer ole.CoUninitialize()

	unknown, err := oleutil.CreateObject("WbemScripting.SWbemLocator")
	if err != nil {
		return err
	} else if unknown == nil {
		return ErrNilCreateObject
	}
	defer unknown.Release()

	wmi, err := unknown.QueryInterface(ole.IID_IDispatch)
	if err != nil {
		return err
	}
	defer wmi.Release()

	// service is a SWbemServices
	serviceRaw, err := oleutil.CallMethod(wmi, "ConnectServer", connectServerArgs...)
	if err != nil {
		return err
	}
	service := serviceRaw.ToIDispatch()
	defer serviceRaw.Clear()

	// result is a SWBemObjectSet
	resultRaw, err := oleutil.CallMethod(service, "ExecQuery", query)
	if err != nil {
		return err
	}
	result := resultRaw.ToIDispatch()
	defer resultRaw.Clear()

	count, err := oleInt64(result, "Count")
	if err != nil {
		return err
	}

	enumProperty, err := result.GetProperty("_NewEnum")
	if err != nil {
		return err
	}
	defer enumProperty.Clear()

	enum, err := enumProperty.ToIUnknown().IEnumVARIANT(ole.IID_IEnumVariant)
	if err != nil {
		return err
	}
	if enum == nil {
		return fmt.Errorf("can't get IEnumVARIANT, enum is nil")
	}
	defer enum.Release()

	// Initialize a slice with Count capacity
	dv.Set(reflect.MakeSlice(dv.Type(), 0, int(count)))

	var errFieldMismatch error
	for itemRaw, length, err := enum.Next(1); length > 0; itemRaw, length, err = enum.Next(1) {
		if err != nil {
			return err
		}

		err := func() error {
			// item is a SWbemObject, but really a Win32_Process
			item := itemRaw.ToIDispatch()
			defer item.Release()

			ev := reflect.New(elemType)
			if err = c.loadEntity(ev.Interface(), item); err != nil {
				if _, ok := err.(*ErrFieldMismatch); ok {
					// We continue loading entities even in the face of field mismatch errors.
					// If we encounter any other error, that other error is returned. Otherwise,
					// an ErrFieldMismatch is returned.
					errFieldMismatch = err
				} else {
					return err
				}
			}
			if mat != multiArgTypeStructPtr {
				ev = ev.Elem()
			}
			dv.Set(reflect.Append(dv, ev))
			return nil
		}()
		if err != nil {
			return err
		}
	}
	return errFieldMismatch
}

// ErrFieldMismatch is returned when a field is to be loaded into a different
// type than the one it was stored from, or when a field is missing or
// unexported in the destination struct.
// StructType is the type of the struct pointed to by the destination argument.
type ErrFieldMismatch struct {
	StructType reflect.Type
	FieldName  string
	Reason     string
}

func (e *ErrFieldMismatch) Error() string {
	return fmt.Sprintf("wmi: cannot load field %q into a %q: %s",
		e.FieldName, e.StructType, e.Reason)
}

var timeType = reflect.TypeOf(time.Time{})

// loadEntity loads a SWbemObject into a struct pointer.
func (c *Client) loadEntity(dst interface{}, src *ole.IDispatch) (errFieldMismatch error) {
	v := reflect.ValueOf(dst).Elem()
	for i := 0; i < v.NumField(); i++ {
		f := v.Field(i)
		of := f
		isPtr := f.Kind() == reflect.Ptr
		if isPtr {
			ptr := reflect.New(f.Type().Elem())
			f.Set(ptr)
			f = f.Elem()
		}
		n := v.Type().Field(i).Name
		if !f.CanSet() {
			return &ErrFieldMismatch{
				StructType: of.Type(),
				FieldName:  n,
				Reason:     "CanSet() is false",
			}
		}
		prop, err := oleutil.GetProperty(src, n)
		if err != nil {
			if !c.AllowMissingFields {
				errFieldMismatch = &ErrFieldMismatch{
					StructType: of.Type(),
					FieldName:  n,
					Reason:     "no such struct field",
				}
			}
			continue
		}
		defer prop.Clear()

		if prop.VT == 0x1 { //VT_NULL
			continue
		}

		switch val := prop.Value().(type) {
		case int8, int16, int32, int64, int:
			v := reflect.ValueOf(val).Int()
			switch f.Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				f.SetInt(v)
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
				f.SetUint(uint64(v))
			default:
				return &ErrFieldMismatch{
					StructType: of.Type(),
					FieldName:  n,
					Reason:     "not an integer class",
				}
			}
		case uint8, uint16, uint32, uint64:
			v := reflect.ValueOf(val).Uint()
			switch f.Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				f.SetInt(int64(v))
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
				f.SetUint(v)
			default:
				return &ErrFieldMismatch{
					StructType: of.Type(),
					FieldName:  n,
					Reason:     "not an integer class",
				}
			}
		case string:
			switch f.Kind() {
			case reflect.String:
				f.SetString(val)
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				iv, err := strconv.ParseInt(val, 10, 64)
				if err != nil {
					return err
				}
				f.SetInt(iv)
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
				uv, err := strconv.ParseUint(val, 10, 64)
				if err != nil {
					return err
				}
				f.SetUint(uv)
			case reflect.Struct:
				switch f.Type() {
				case timeType:
					if len(val) == 25 {
						mins, err := strconv.Atoi(val[22:])
						if err != nil {
							return err
						}
						val = val[:22] + fmt.Sprintf("%02d%02d", mins/60, mins%60)
					}
					t, err := time.Parse("20060102150405.000000-0700", val)
					if err != nil {
						return err
					}
					f.Set(reflect.ValueOf(t))
				}
			}
		case bool:
			switch f.Kind() {
			case reflect.Bool:
				f.SetBool(val)
			default:
				return &ErrFieldMismatch{
					StructType: of.Type(),
					FieldName:  n,
					Reason:     "not a bool",
				}
			}
		case float32:
			switch f.Kind() {
			case reflect.Float32:
				f.SetFloat(float64(val))
			default:
				return &ErrFieldMismatch{
					StructType: of.Type(),
					FieldName:  n,
					Reason:     "not a Float32",
				}
			}
		default:
			if f.Kind() == reflect.Slice {
				switch f.Type().Elem().Kind() {
				case reflect.String:
					safeArray := prop.ToArray()
					if safeArray != nil {
						arr := safeArray.ToValueArray()
						fArr := reflect.MakeSlice(f.Type(), len(arr), len(arr))
						for i, v := range arr {
							s := fArr.Index(i)
							s.SetString(v.(string))
						}
						f.Set(fArr)
					}
				case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uint:
					safeArray := prop.ToArray()
					if safeArray != nil {
						arr := safeArray.ToValueArray()
						fArr := reflect.MakeSlice(f.Type(), len(arr), len(arr))
						for i, v := range arr {
							s := fArr.Index(i)
							s.SetUint(reflect.ValueOf(v).Uint())
						}
						f.Set(fArr)
					}
				case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Int:
					safeArray := prop.ToArray()
					if safeArray != nil {
						arr := safeArray.ToValueArray()
						fArr := reflect.MakeSlice(f.Type(), len(arr), len(arr))
						for i, v := range arr {
							s := fArr.Index(i)
							s.SetInt(reflect.ValueOf(v).Int())
						}
						f.Set(fArr)
					}
				default:
					return &ErrFieldMismatch{
						StructType: of.Type(),
						FieldName:  n,
						Reason:     fmt.Sprintf("unsupported slice type (%T)", val),
					}
				}
			} else {
				typeof := reflect.TypeOf(val)
				if typeof == nil && (isPtr || c.NonePtrZero) {
					if (isPtr && c.PtrNil) || (!isPtr && c.NonePtrZero) {
						of.Set(reflect.Zero(of.Type()))
					}
					break
				}
				return &ErrFieldMismatch{
					StructType: of.Type(),
					FieldName:  n,
					Reason:     fmt.Sprintf("unsupported type (%T)", val),
				}
			}
		}
	}
	return errFieldMismatch
}

type multiArgType int

const (
	multiArgTypeInvalid multiArgType = iota
	multiArgTypeStruct
	multiArgTypeStructPtr
)

// checkMultiArg checks that v has type []S, []*S for some struct type S.
//
// It returns what category the slice's elements are, and the reflect.Type
// that represents S.
func checkMultiArg(v reflect.Value) (m multiArgType, elemType reflect.Type) {
	if v.Kind() != reflect.Slice {
		return multiArgTypeInvalid, nil
	}
	elemType = v.Type().Elem()
	switch elemType.Kind() {
	case reflect.Struct:
		return multiArgTypeStruct, elemType
	case reflect.Ptr:
		elemType = elemType.Elem()
		if elemType.Kind() == reflect.Struct {
			return multiArgTypeStructPtr, elemType
		}
	}
	return multiArgTypeInvalid, nil
}

func oleInt64(item *ole.IDispatch, prop string) (int64, error) {
	v, err := oleutil.GetProperty(item, prop)
	if err != nil {
		return 0, err
	}
	defer v.Clear()

	i := int64(v.Val)
	return i, nil
}

// CreateQuery returns a WQL query string that queries all columns of src. where
// is an optional string that is appended to the query, to be used with WHERE
// clauses. In such a case, the "WHERE" string should appear at the beginning.
func CreateQuery(src interface{}, where string) string {
	var b bytes.Buffer
	b.WriteString("SELECT ")
	s := reflect.Indirect(reflect.ValueOf(src))
	t := s.Type()
	if s.Kind() == reflect.Slice {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return ""
	}
	var fields []string
	for i := 0; i < t.NumField(); i++ {
		fields = append(fields, t.Field(i).Name)
	}
	b.WriteString(strings.Join(fields, ", "))
	b.WriteString(" FROM ")
	b.WriteString(t.Name())
	b.WriteString(" " + where)
	return b.String()
}
