package dbus

import (
	"bytes"
	"reflect"
	"strings"
	"sync"
)

func newIntrospectIntf(h *defaultHandler) *exportedIntf {
	methods := make(map[string]Method)
	methods["Introspect"] = exportedMethod{
		reflect.ValueOf(func(msg Message) (string, *Error) {
			path := msg.Headers[FieldPath].value.(ObjectPath)
			return h.introspectPath(path), nil
		}),
	}
	return newExportedIntf(methods, true)
}

func newDefaultHandler() *defaultHandler {
	h := &defaultHandler{
		objects:     make(map[ObjectPath]*exportedObj),
		defaultIntf: make(map[string]*exportedIntf),
	}
	h.defaultIntf["org.freedesktop.DBus.Introspectable"] = newIntrospectIntf(h)
	return h
}

type defaultHandler struct {
	sync.RWMutex
	objects     map[ObjectPath]*exportedObj
	defaultIntf map[string]*exportedIntf
}

func (h *defaultHandler) PathExists(path ObjectPath) bool {
	_, ok := h.objects[path]
	return ok
}

func (h *defaultHandler) introspectPath(path ObjectPath) string {
	subpath := make(map[string]struct{})
	var xml bytes.Buffer
	xml.WriteString("<node>")
	for obj, _ := range h.objects {
		p := string(path)
		if p != "/" {
			p += "/"
		}
		if strings.HasPrefix(string(obj), p) {
			node_name := strings.Split(string(obj[len(p):]), "/")[0]
			subpath[node_name] = struct{}{}
		}
	}
	for s, _ := range subpath {
		xml.WriteString("\n\t<node name=\"" + s + "\"/>")
	}
	xml.WriteString("\n</node>")
	return xml.String()
}

func (h *defaultHandler) LookupObject(path ObjectPath) (ServerObject, bool) {
	h.RLock()
	defer h.RUnlock()
	object, ok := h.objects[path]
	if ok {
		return object, ok
	}

	// If an object wasn't found for this exact path,
	// look for a matching subtree registration
	subtreeObject := newExportedObject()
	path = path[:strings.LastIndex(string(path), "/")]
	for len(path) > 0 {
		object, ok = h.objects[path]
		if ok {
			for name, iface := range object.interfaces {
				// Only include this handler if it registered for the subtree
				if iface.isFallbackInterface() {
					subtreeObject.interfaces[name] = iface
				}
			}
			break
		}

		path = path[:strings.LastIndex(string(path), "/")]
	}

	for name, intf := range h.defaultIntf {
		if _, exists := subtreeObject.interfaces[name]; exists {
			continue
		}
		subtreeObject.interfaces[name] = intf
	}

	return subtreeObject, true
}

func (h *defaultHandler) AddObject(path ObjectPath, object *exportedObj) {
	h.Lock()
	h.objects[path] = object
	h.Unlock()
}

func (h *defaultHandler) DeleteObject(path ObjectPath) {
	h.Lock()
	delete(h.objects, path)
	h.Unlock()
}

type exportedMethod struct {
	reflect.Value
}

func (m exportedMethod) Call(args ...interface{}) ([]interface{}, error) {
	t := m.Type()

	params := make([]reflect.Value, len(args))
	for i := 0; i < len(args); i++ {
		params[i] = reflect.ValueOf(args[i]).Elem()
	}

	ret := m.Value.Call(params)

	err := ret[t.NumOut()-1].Interface().(*Error)
	ret = ret[:t.NumOut()-1]
	out := make([]interface{}, len(ret))
	for i, val := range ret {
		out[i] = val.Interface()
	}
	if err == nil {
		//concrete type to interface nil is a special case
		return out, nil
	}
	return out, err
}

func (m exportedMethod) NumArguments() int {
	return m.Value.Type().NumIn()
}

func (m exportedMethod) ArgumentValue(i int) interface{} {
	return reflect.Zero(m.Type().In(i)).Interface()
}

func (m exportedMethod) NumReturns() int {
	return m.Value.Type().NumOut()
}

func (m exportedMethod) ReturnValue(i int) interface{} {
	return reflect.Zero(m.Type().Out(i)).Interface()
}

func newExportedObject() *exportedObj {
	return &exportedObj{
		interfaces: make(map[string]*exportedIntf),
	}
}

type exportedObj struct {
	interfaces map[string]*exportedIntf
}

func (obj *exportedObj) LookupInterface(name string) (Interface, bool) {
	if name == "" {
		return obj, true
	}
	intf, exists := obj.interfaces[name]
	return intf, exists
}

func (obj *exportedObj) AddInterface(name string, iface *exportedIntf) {
	obj.interfaces[name] = iface
}

func (obj *exportedObj) DeleteInterface(name string) {
	delete(obj.interfaces, name)
}

func (obj *exportedObj) LookupMethod(name string) (Method, bool) {
	for _, intf := range obj.interfaces {
		method, exists := intf.LookupMethod(name)
		if exists {
			return method, exists
		}
	}
	return nil, false
}

func (obj *exportedObj) isFallbackInterface() bool {
	return false
}

func newExportedIntf(methods map[string]Method, includeSubtree bool) *exportedIntf {
	return &exportedIntf{
		methods:        methods,
		includeSubtree: includeSubtree,
	}
}

type exportedIntf struct {
	methods map[string]Method

	// Whether or not this export is for the entire subtree
	includeSubtree bool
}

func (obj *exportedIntf) LookupMethod(name string) (Method, bool) {
	out, exists := obj.methods[name]
	return out, exists
}

func (obj *exportedIntf) isFallbackInterface() bool {
	return obj.includeSubtree
}

func newDefaultSignalHandler() *defaultSignalHandler {
	return &defaultSignalHandler{}
}

func isDefaultSignalHandler(handler SignalHandler) bool {
	_, ok := handler.(*defaultSignalHandler)
	return ok
}

type defaultSignalHandler struct {
	sync.RWMutex
	closed  bool
	signals []chan<- *Signal
}

func (sh *defaultSignalHandler) DeliverSignal(intf, name string, signal *Signal) {
	sh.RLock()
	defer sh.RUnlock()
	if sh.closed {
		return
	}
	for _, ch := range sh.signals {
		ch <- signal
	}
}

func (sh *defaultSignalHandler) Init() error {
	sh.Lock()
	sh.signals = make([]chan<- *Signal, 0)
	sh.Unlock()
	return nil
}

func (sh *defaultSignalHandler) Terminate() {
	sh.Lock()
	sh.closed = true
	for _, ch := range sh.signals {
		close(ch)
	}
	sh.signals = nil
	sh.Unlock()
}

func (sh *defaultSignalHandler) addSignal(ch chan<- *Signal) {
	sh.Lock()
	defer sh.Unlock()
	if sh.closed {
		return
	}
	sh.signals = append(sh.signals, ch)

}

func (sh *defaultSignalHandler) removeSignal(ch chan<- *Signal) {
	sh.Lock()
	defer sh.Unlock()
	if sh.closed {
		return
	}
	for i := len(sh.signals) - 1; i >= 0; i-- {
		if ch == sh.signals[i] {
			copy(sh.signals[i:], sh.signals[i+1:])
			sh.signals[len(sh.signals)-1] = nil
			sh.signals = sh.signals[:len(sh.signals)-1]
		}
	}
}
