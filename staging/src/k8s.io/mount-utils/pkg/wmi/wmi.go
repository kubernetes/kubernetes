//go:build windows
// +build windows

/*
Copyright 2025 The Kubernetes Authors.

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

package wmi

import (
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/go-ole/go-ole"
	"github.com/go-ole/go-ole/oleutil"
	"golang.org/x/sys/windows"
	"k8s.io/klog/v2"
)

const (
	WMINamespaceCimV2   = "Root\\CimV2"
	WMINamespaceStorage = "Root\\Microsoft\\Windows\\Storage"
	WMINamespaceSmb     = "Root\\Microsoft\\Windows\\Smb"

	WBEM_S_FALSE = 0x00000001

	// rpcETooLate is returned when CoInitializeSecurity has already been called.
	rpcETooLate = 0x80010119

	envWMIComSecurity = "WMI_COM_SECURITY"
)

var initCOMSecurity sync.Once

var (
	ErrNotFound      = errors.New("not found")
	ErrStopIteration = errors.New("stop iteration")

	DiscardOutputParameter = func(_ string, _ *ole.VARIANT) (interface{}, error) {
		return nil, nil
	}
	DiscardResult = func(_ *ole.VARIANT) error {
		return nil
	}
)

// IsNotFound returns true if it's a "not found" error.
func IsNotFound(err error) bool {
	return errors.Is(err, ErrNotFound)
}

// IgnoreNotFound returns nil if the error is nil or a "not found" error,
// otherwise returns the original error.
func IgnoreNotFound(err error) error {
	if err == nil || IsNotFound(err) {
		return nil
	}
	return err
}

func IsEndOfEnum(err error) bool {
	var oleErr *ole.OleError
	return errors.As(err, &oleErr) && oleErr.Code() == WBEM_S_FALSE
}

// QueryBuilder is a builder for WMI queries.
type QueryBuilder struct {
	Class      string
	Namespace  string
	Selectors  []string
	Conditions []Condition
}

// Condition is a condition for a WMI query.
type Condition struct {
	Field    string
	Operator string
	Value    any
}

func (c Condition) String() string {
	return fmt.Sprintf("%s %s %s", c.Field, c.Operator, formatValue(c.Value))
}

// WithCondition creates a new condition for a WMI query.
func WithCondition(field string, operator string, value any) Condition {
	return Condition{Field: field, Operator: operator, Value: value}
}

// NewQuery creates a new QueryBuilder for a WMI class.
func NewQuery(class string) *QueryBuilder {
	return &QueryBuilder{
		Class:     class,
		Namespace: WMINamespaceCimV2, // default, override if needed
	}
}

func (q *QueryBuilder) WithNamespace(ns string) *QueryBuilder {
	q.Namespace = ns
	return q
}

func (q *QueryBuilder) Select(selectors ...string) *QueryBuilder {
	q.Selectors = append(q.Selectors, selectors...)
	return q
}

func (q *QueryBuilder) WithCondition(field string, operator string, value any) *QueryBuilder {
	q.Conditions = append(q.Conditions, Condition{Field: field, Operator: operator, Value: value})
	return q
}

func (q *QueryBuilder) WithConditions(conditions ...Condition) *QueryBuilder {
	q.Conditions = append(q.Conditions, conditions...)
	return q
}

func (q *QueryBuilder) buildWhereClause() string {
	if len(q.Conditions) == 0 {
		return ""
	}

	conditions := make([]string, len(q.Conditions))
	for i, condition := range q.Conditions {
		conditions[i] = condition.String()
	}

	return "WHERE " + strings.Join(conditions, " AND ")
}

func (q *QueryBuilder) Build() string {
	selectPart := "*"

	if len(q.Selectors) > 0 {
		selectPart = strings.Join(q.Selectors, ", ")
	}

	query := "SELECT " + selectPart + " FROM " + q.Class

	where := q.buildWhereClause()
	if where != "" {
		query += " " + where
	}

	return query
}

func formatValue(v any) string {
	switch x := v.(type) {

	case string:
		// WQL escaping: double single quotes and escape backslashes
		escaped := strings.ReplaceAll(x, "'", "''")
		escaped = strings.ReplaceAll(escaped, "\\", "\\\\")
		return "'" + escaped + "'"

	case int, int32, int64, uint, uint32, uint64:
		return fmt.Sprintf("%v", x)

	case bool:
		if x {
			return "TRUE"
		}
		return "FALSE"

	default:
		return fmt.Sprintf("'%v'", x)
	}
}

func comSecurityEnabled() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(envWMIComSecurity)))
	switch v {
	case "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

// InitializeCOMSecurity registers process-wide COM security for WMI/DCOM.
//
// It is enabled by default and runs at most once per process (including from init).
// Set WMI_COM_SECURITY to false/0/no/off to skip. Errors (including missing permissions or an already
// initialized process) are logged and ignored; this function never panics.
func InitializeCOMSecurity() {
	initCOMSecurity.Do(func() {
		if !comSecurityEnabled() {
			klog.V(6).Infof("COM security initialization disabled via %s", envWMIComSecurity)
			return
		}

		err := ole.CoInitializeSecurity(-1, 0, 3, 0)
		if err == nil {
			klog.V(10).Infof("COM security initialized for WMI/DCOM")
			return
		}

		var oleErr *ole.OleError
		if errors.As(err, &oleErr) && oleErr.Code() == rpcETooLate {
			klog.V(10).Infof("COM security already initialized for this process")
			return
		}

		klog.V(4).Infof("CoInitializeSecurity failed (non-fatal): %v", err)
	})
}

func init() {
	InitializeCOMSecurity()
}

// WithCOMThread runs the given function `fn` on a locked OS thread
// with COM initialized using COINIT_MULTITHREADED.
//
// This is necessary for using COM/OLE APIs directly (e.g., via go-ole),
// because COM requires that initialization and usage occur on the same thread.
//
// It performs the following steps:
//   - Locks the current goroutine to its OS thread
//   - Calls ole.CoInitializeEx with COINIT_MULTITHREADED
//   - Executes the user-provided function
//   - Uninitializes COM
//   - Unlocks the thread
//
// Process-wide COM security is initialized once at package init (best-effort).
//
// If COM initialization fails, or if the user's function returns an error,
// that error is returned by WithCOMThread.
func WithCOMThread(fn func() error) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := ole.CoInitializeEx(0, ole.COINIT_MULTITHREADED); err != nil {
		var oleError *ole.OleError
		if errors.As(err, &oleError) && oleError != nil && oleError.Code() == uintptr(windows.S_FALSE) {
			klog.V(10).Infof("COM library has been already initialized for the calling thread, proceeding to the function with no error")
			err = nil
		}
		if err != nil {
			return err
		}
	} else {
		klog.V(10).Infof("COM library is initialized for the calling thread")
	}
	defer ole.CoUninitialize()

	return fn()
}

// WithWMIService runs the given function with a WMI service.
//
// It creates a new WMI service and calls the given function with it.
// The callback fn receives an *ole.IDispatch that is only valid for the duration
// of the callback; callers must NOT store or use it after fn returns.
func WithWMIService(namespace string, fn func(*ole.IDispatch) error) error {
	locatorUnknown, err := oleutil.CreateObject("WbemScripting.SWbemLocator")
	if err != nil {
		return err
	}
	defer locatorUnknown.Release()

	locator, err := locatorUnknown.QueryInterface(ole.IID_IDispatch)
	if err != nil {
		return err
	}
	defer locator.Release()

	serviceRaw, err := oleutil.CallMethod(locator, "ConnectServer", nil, namespace)
	if err != nil {
		return err
	}
	defer serviceRaw.Clear()
	// serviceRaw.Clear() releases the underlying IDispatch via VariantClear;
	// do NOT call service.Release() separately (double-release).
	service := serviceRaw.ToIDispatch()

	return fn(service)
}

// Query queries the WMI objects with the given namespace and query.
// Callers must ensure COM is initialized on the current OS thread (e.g., via WithCOMThread).
// The callback fn receives an *ole.IDispatch that is only valid for the duration
// of the callback; callers must NOT store or use it after fn returns.
func Query(namespace, query string, fn func(item *ole.IDispatch) error) error {
	err := WithWMIService(namespace, func(service *ole.IDispatch) error {
		resultRaw, err := oleutil.CallMethod(service, "ExecQuery", query)
		if err != nil {
			klog.V(4).Infof("ExecQuery: (namespace: %s, query: %s), error: %v", namespace, query, err)
			return err
		}
		defer resultRaw.Clear()
		// resultRaw.Clear() releases the underlying IDispatch via VariantClear;
		// do NOT call result.Release() separately (double-release).
		result := resultRaw.ToIDispatch()

		klog.V(10).Infof("ExecQuery: (namespace: %s, query: %s) -> enumerating results", namespace, query)

		err = Enumerate(result, func(item *ole.VARIANT) error {
			return fn(item.ToIDispatch())
		})
		if err != nil {
			if errors.Is(err, ErrStopIteration) {
				return nil // stop early
			}

			return err
		}

		return nil
	})
	return err
}

func QueryWithBuilder(builder *QueryBuilder, fn func(item *ole.IDispatch) error) error {
	return Query(builder.Namespace, builder.Build(), fn)
}

// Associators gets the associators of the given object.
func Associators(obj *ole.IDispatch, assocClass, resultClass, role, resultRole string, fn func(*ole.IDispatch) error) error {
	klog.V(10).Infof("Associators: obj: %v, assocClass: %s, resultClass: %s, role: %s, resultRole: %s", obj, assocClass, resultClass, role, resultRole)

	// Associators_ is the key WMI COM method
	result, err := oleutil.CallMethod(
		obj,
		"Associators_",
		assocClass,  // assocClass
		resultClass, // resultClass
		role,        // role
		resultRole,  // resultRole
	)
	if err != nil {
		return fmt.Errorf("failed to list associators. error: %w", err)
	}

	defer result.Clear()

	err = Enumerate(result.ToIDispatch(), func(item *ole.VARIANT) error {
		return fn(item.ToIDispatch())
	})

	return err
}

// Enumerate enumerates the items of the given object.
//
// Doc: https://docs.microsoft.com/en-us/previous-versions/windows/desktop/automat/dispid-constants
func Enumerate(obj *ole.IDispatch, fn func(item *ole.VARIANT) error) error {
	// --- Get _NewEnum ---
	enumProp, err := obj.GetProperty("_NewEnum")
	if err != nil {
		return fmt.Errorf("failed to get _NewEnum from IDispatch %v: %w", obj, err)
	}
	defer enumProp.Clear()

	// --- Get IEnumVARIANT ---
	enum, err := enumProp.ToIUnknown().IEnumVARIANT(ole.IID_IEnumVariant)
	if err != nil {
		return fmt.Errorf("failed to get IEnumVARIANT from IDispatch %v: %w", obj, err)
	}
	if enum == nil {
		return fmt.Errorf("IEnumVARIANT is nil for IDispatch %v", obj)
	}
	defer enum.Release()

	for {
		item, length, err := enum.Next(1)
		if err != nil {
			if IsEndOfEnum(err) {
				break
			}
			return err
		}
		if length == 0 {
			break
		}

		err = func() error {
			defer item.Clear()
			return fn(&item)
		}()
		if err != nil {
			return err
		}
	}

	return nil
}

// CallMethodOnWMIClass calls a method on a WMI class and returns the return value.
func CallMethodOnWMIClass(namespace, class, methodName string, input map[string]interface{}, handler func(string, *ole.VARIANT) (interface{}, error)) (uint32, map[string]interface{}, error) {
	var returnValue uint32
	var output map[string]interface{}
	err := WithWMIService(namespace, func(service *ole.IDispatch) error {
		classRaw, err := oleutil.CallMethod(service, "Get", class)
		if err != nil {
			return fmt.Errorf("get class %s failed: %w", class, err)
		}
		defer classRaw.Clear()
		classInst := classRaw.ToIDispatch() // Release is not required as no AddRef is called

		methodsRaw, err := oleutil.GetProperty(classInst, "Methods_")
		if err != nil {
			return err
		}
		defer methodsRaw.Clear()

		methodRaw, err := methodsRaw.ToIDispatch().CallMethod("Item", methodName)
		if err != nil {
			return fmt.Errorf("method %s not found: %w", methodName, err)
		}
		defer methodRaw.Clear()

		params := []interface{}{methodName}

		if len(input) > 0 {
			inParamsRaw, err := methodRaw.ToIDispatch().GetProperty("InParameters")
			if err != nil {
				return err
			}
			defer inParamsRaw.Clear()

			if inParamsRaw.Val != 0 {
				inParamsInstRaw, err := oleutil.CallMethod(inParamsRaw.ToIDispatch(), "SpawnInstance_")
				if err != nil {
					return err
				}
				defer inParamsInstRaw.Clear()

				inParamsInst := inParamsInstRaw.ToIDispatch()

				for k, v := range input {
					putResult, err := oleutil.PutProperty(inParamsInst, k, v)
					if err != nil {
						return fmt.Errorf("set param %s failed: %w", k, err)
					}
					putResult.Clear()
				}

				params = append(params, inParamsInst)
			}
		}

		outParamsRaw, err := classInst.CallMethod("ExecMethod_", params...)
		if err != nil {
			return fmt.Errorf("ExecMethod_ failed: %w", err)
		}
		defer outParamsRaw.Clear()

		outParams := outParamsRaw.ToIDispatch() // Release is not required as no AddRef is called

		rv, err := outParams.GetProperty("ReturnValue")
		if err != nil {
			return fmt.Errorf("ReturnValue failed: %w", err)
		}
		defer rv.Clear()
		returnValue = NewSafeVariant(rv).Uint32()

		propsRaw, err := outParams.GetProperty("Properties_")
		if err != nil {
			return err
		}
		defer propsRaw.Clear()

		props := propsRaw.ToIDispatch()

		output = make(map[string]interface{})
		err = Enumerate(props, func(item *ole.VARIANT) error {
			prop := item.ToIDispatch() // Release is not required as no AddRef is called

			nameVar, err := prop.GetProperty("Name")
			if err != nil {
				return err
			}
			defer nameVar.Clear()

			name := fmt.Sprintf("%v", nameVar.Value())

			valVar, err := prop.GetProperty("Value")
			if err != nil {
				return err
			}
			defer valVar.Clear()

			val, err := handler(name, valVar)
			if err != nil {
				return err
			}
			if val != nil {
				output[name] = val
			}

			return nil
		})
		return err
	})
	return returnValue, output, err
}

// COMDispatchObject is a wrapper around an ole.IDispatch object.
type COMDispatchObject struct {
	obj *ole.IDispatch
}

// NewCOMDispatchObject clones an ole.IDispatch object and returns a COMDispatchObject.
//
// Ownership of the object is not transferred to the COMDispatchObject.
// Instead, it holds a new reference to the object.
func NewCOMDispatchObject(obj *ole.IDispatch) *COMDispatchObject {
	if obj != nil {
		obj.AddRef()
	}
	return &COMDispatchObject{obj: obj}
}

func (c *COMDispatchObject) Dispatch() *ole.IDispatch {
	return c.obj
}

func (c *COMDispatchObject) GetPropertyWithHandler(name string, fn func(*ole.VARIANT) error) error {
	v, err := oleutil.GetProperty(c.Dispatch(), name)
	if err != nil {
		return fmt.Errorf("failed to get property %s: %w", name, err)
	}
	defer v.Clear()

	return fn(v)
}

func (c *COMDispatchObject) GetStringProperty(name string) (string, error) {
	var result string
	err := c.GetPropertyWithHandler(name, func(v *ole.VARIANT) error {
		result = NewSafeVariant(v).String()
		return nil
	})
	if err != nil {
		return "", err
	}
	return result, nil
}

func (c *COMDispatchObject) GetUint32Property(name string) (uint32, error) {
	var result uint32
	err := c.GetPropertyWithHandler(name, func(v *ole.VARIANT) error {
		result = NewSafeVariant(v).Uint32()
		return nil
	})
	if err != nil {
		return 0, err
	}
	return result, nil
}

func (c *COMDispatchObject) GetUint16Property(name string) (uint16, error) {
	var result uint16
	err := c.GetPropertyWithHandler(name, func(v *ole.VARIANT) error {
		result = NewSafeVariant(v).Uint16()
		return nil
	})
	if err != nil {
		return 0, err
	}
	return result, nil
}

func (c *COMDispatchObject) GetBoolProperty(name string) (bool, error) {
	var result bool
	err := c.GetPropertyWithHandler(name, func(v *ole.VARIANT) error {
		result = NewSafeVariant(v).Bool()
		return nil
	})
	if err != nil {
		return false, err
	}
	return result, nil
}

func (c *COMDispatchObject) GetStringPropertyAsUint64(name string) (uint64, error) {
	str, err := c.GetStringProperty(name)
	if err != nil {
		return 0, err
	}
	if str == "" {
		return 0, fmt.Errorf("WMI property %q is empty; cannot parse as uint64", name)
	}
	result, err := strconv.ParseUint(str, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse WMI property %q value %q as uint64: %w", name, str, err)
	}
	return result, nil
}

func (c *COMDispatchObject) CallMethod(name string, fn func(*ole.VARIANT) error, params ...interface{}) error {
	v, err := oleutil.CallMethod(c.Dispatch(), name, params...)
	if err != nil {
		return err
	}
	defer v.Clear()

	return fn(v)
}

func (c *COMDispatchObject) CallVoid(name string, params ...interface{}) error {
	return c.CallMethod(name, DiscardResult, params...)
}

func (c *COMDispatchObject) CallUint32(name string, params ...interface{}) (uint32, error) {
	var result uint32
	err := c.CallMethod(name, func(variant *ole.VARIANT) error {
		result = NewSafeVariant(variant).Uint32()
		return nil
	}, params...)
	return result, err
}

func (c *COMDispatchObject) GetAssociated(scope *Scope, assocClass, resultClass, role, resultRole string) ([]*COMDispatchObject, error) {
	results := make([]*COMDispatchObject, 0)

	err := Associators(c.Dispatch(), assocClass, resultClass, role, resultRole, func(item *ole.IDispatch) error {
		obj := NewCOMDispatchObject(item)
		results = append(results, scope.Track(obj))
		return nil
	})
	if err != nil {
		return nil, err
	}

	return results, nil
}

func (c *COMDispatchObject) Clone() *COMDispatchObject {
	return NewCOMDispatchObject(c.obj)
}

func (c *COMDispatchObject) release() {
	if c.obj != nil {
		c.obj.Release()
		c.obj = nil
	}
}

// QueryObjectsWithBuilder queries WMI objects with the given query builder.
func QueryObjectsWithBuilder(scope *Scope, q *QueryBuilder) ([]*COMDispatchObject, error) {
	results := make([]*COMDispatchObject, 0)

	err := Query(q.Namespace, q.Build(), func(item *ole.IDispatch) error {
		obj := NewCOMDispatchObject(item)
		results = append(results, scope.Track(obj))
		return nil
	})
	if err != nil {
		return nil, err
	}

	return results, nil
}

// QueryFirstObjectWithBuilder queries the first object with the given query builder.
//
// It returns the object and the error.
//
// If the object is not found, it returns ErrNotFound.
//
// If the query fails, it returns the error.
//
// If the query is successful, it returns the object.
func QueryFirstObjectWithBuilder(scope *Scope, q *QueryBuilder) (*COMDispatchObject, error) {
	var result *COMDispatchObject

	err := Query(q.Namespace, q.Build(), func(item *ole.IDispatch) error {
		result = scope.Track(NewCOMDispatchObject(item))

		// stop early
		return ErrStopIteration
	})

	if err != nil {
		return nil, err
	}

	if result == nil {
		return nil, ErrNotFound
	}

	return result, nil
}

func ForEach(items []*COMDispatchObject, fn func(*COMDispatchObject) error) error {
	for _, item := range items {
		if err := fn(item); err != nil {
			if errors.Is(err, ErrStopIteration) {
				return nil // stop early
			}
			return err
		}
	}
	return nil
}

// SafeVariant is a wrapper around an ole.VARIANT object.
type SafeVariant struct {
	v *ole.VARIANT
}

func NewSafeVariant(v *ole.VARIANT) SafeVariant {
	return SafeVariant{v: v}
}

func (s SafeVariant) Bool() bool {
	if s.v == nil {
		return false
	}
	if s.v.VT == ole.VT_BOOL {
		return (s.v.Val & 0xffff) != 0
	}
	return false
}

func (s SafeVariant) String() string {
	if s.v == nil {
		return ""
	}
	return s.v.ToString()
}

func isInteger(vt ole.VT) bool {
	switch vt {
	case ole.VT_I1, ole.VT_UI1, ole.VT_I2, ole.VT_UI2, ole.VT_I4, ole.VT_UI4, ole.VT_I8, ole.VT_UI8, ole.VT_INT, ole.VT_UINT:
		return true
	}
	return false
}

func isUnsigned(vt ole.VT) bool {
	switch vt {
	case ole.VT_UI1, ole.VT_UI2, ole.VT_UI4, ole.VT_UI8, ole.VT_UINT:
		return true
	}
	return false
}

func (s SafeVariant) Int() int {
	if s.v == nil || !isInteger(s.v.VT) {
		return 0
	}
	// For unsigned types with high bit set, clamp to max int to avoid
	// negative values from sign interpretation.
	if isUnsigned(s.v.VT) && (s.v.Val < 0 || s.v.Val > math.MaxInt) {
		return math.MaxInt
	}
	return int(s.v.Val)
}

func (s SafeVariant) Int32() int32 {
	if s.v == nil || !isInteger(s.v.VT) {
		return 0
	}
	if isUnsigned(s.v.VT) && (s.v.Val < 0 || s.v.Val > math.MaxInt32) {
		return math.MaxInt32
	}
	return int32(s.v.Val)
}

func (s SafeVariant) Int64() int64 {
	if s.v == nil || !isInteger(s.v.VT) {
		return 0
	}
	return s.v.Val
}

func (s SafeVariant) Uint16() uint16 {
	if s.v == nil || !isInteger(s.v.VT) {
		return 0
	}
	return uint16(s.v.Val & 0xFFFF)
}

func (s SafeVariant) Uint32() uint32 {
	if s.v == nil || !isInteger(s.v.VT) {
		return 0
	}
	return uint32(s.v.Val & 0xFFFFFFFF)
}

func (s SafeVariant) Uint64() uint64 {
	if s.v == nil || !isInteger(s.v.VT) {
		return 0
	}
	return uint64(s.v.Val)
}

func (s SafeVariant) Raw() (interface{}, bool) {
	if s.v == nil {
		return nil, false
	}
	return s.v.Value(), true
}

func (s SafeVariant) Value() (interface{}, error) {
	if s.v == nil {
		return nil, fmt.Errorf("variant is nil")
	}
	return s.v.Value(), nil
}

func (s SafeVariant) Clear() error {
	if s.v == nil {
		return nil
	}

	return s.v.Clear()
}

// Scope is a scope for a WMI operation to track the objects created during the operation.
//
// It is used to ensure that the objects are released when the scope is closed.
type Scope struct {
	objs []*COMDispatchObject
}

// Track tracks the given object in the scope.
//
// It returns the object itself.
func (s *Scope) Track(obj *COMDispatchObject) *COMDispatchObject {
	if obj == nil {
		return nil
	}

	s.objs = append(s.objs, obj)
	return obj
}

// Close closes the scope and releases the objects.
func (s *Scope) Close() {
	for i := len(s.objs) - 1; i >= 0; i-- {
		if s.objs[i] != nil {
			s.objs[i].release()
		}
	}
	s.objs = nil
}

// WithScope runs the given function with a new scope.
// It returns the error from the function.
func WithScope(fn func(*Scope) error) error {
	scope := &Scope{}
	defer scope.Close()

	return fn(scope)
}

// WMIError is an error for a WMI operation.
type WMIError struct {
	Method  string
	Class   string
	Target  string
	Code    uint32
	Details string
}

// NewWMIError creates a new WMIError.
func NewWMIError(class, method string, target *ole.IDispatch, code uint32) *WMIError {
	return &WMIError{
		Class:  class,
		Method: method,
		Target: fmt.Sprintf("%v", target),
		Code:   code,
	}
}

func (e *WMIError) Error() string {
	if e.Details != "" {
		return fmt.Sprintf("WMI %s.%s failed (target=%v): %s (code=%d)", e.Class, e.Method, e.Target, e.Details, e.Code)
	}
	return fmt.Sprintf("WMI %s.%s failed (target=%v): code=%d", e.Class, e.Method, e.Target, e.Code)
}
