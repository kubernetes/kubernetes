/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package util

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"github.com/google/gofuzz"
)

// For testing, bypass HandleCrash.
var ReallyCrash bool

// PanicHandlers is a list of functions which will be invoked when a panic happens.
var PanicHandlers = []func(interface{}){logPanic}

// HandleCrash simply catches a crash and logs an error. Meant to be called via defer.
func HandleCrash() {
	if ReallyCrash {
		return
	}
	if r := recover(); r != nil {
		for _, fn := range PanicHandlers {
			fn(r)
		}
	}
}

// logPanic logs the caller tree when a panic occurs.
func logPanic(r interface{}) {
	callers := ""
	for i := 0; true; i++ {
		_, file, line, ok := runtime.Caller(i)
		if !ok {
			break
		}
		callers = callers + fmt.Sprintf("%v:%v\n", file, line)
	}
	glog.Errorf("Recovered from panic: %#v (%v)\n%v", r, r, callers)
}

// ErrorHandlers is a list of functions which will be invoked when an unreturnable
// error occurs.
var ErrorHandlers = []func(error){logError}

// HandlerError is a method to invoke when a non-user facing piece of code cannot
// return an error and needs to indicate it has been ignored. Invoking this method
// is preferable to logging the error - the default behavior is to log but the
// errors may be sent to a remote server for analysis.
func HandleError(err error) {
	for _, fn := range ErrorHandlers {
		fn(err)
	}
}

// logError prints an error with the call stack of the location it was reported
func logError(err error) {
	glog.ErrorDepth(2, err)
}

// Forever loops forever running f every period.  Catches any panics, and keeps going.
// Deprecated. Please use Until and pass NeverStop as the stopCh.
func Forever(f func(), period time.Duration) {
	Until(f, period, nil)
}

// NeverStop may be passed to Until to make it never stop.
var NeverStop <-chan struct{} = make(chan struct{})

// Until loops until stop channel is closed, running f every period.
// Catches any panics, and keeps going. f may not be invoked if
// stop channel is already closed.
func Until(f func(), period time.Duration, stopCh <-chan struct{}) {
	for {
		select {
		case <-stopCh:
			return
		default:
		}
		func() {
			defer HandleCrash()
			f()
		}()
		time.Sleep(period)
	}
}

// IntOrString is a type that can hold an int or a string.  When used in
// JSON or YAML marshalling and unmarshalling, it produces or consumes the
// inner type.  This allows you to have, for example, a JSON field that can
// accept a name or number.
type IntOrString struct {
	Kind   IntstrKind
	IntVal int
	StrVal string
}

// IntstrKind represents the stored type of IntOrString.
type IntstrKind int

const (
	IntstrInt    IntstrKind = iota // The IntOrString holds an int.
	IntstrString                   // The IntOrString holds a string.
)

// NewIntOrStringFromInt creates an IntOrString object with an int value.
func NewIntOrStringFromInt(val int) IntOrString {
	return IntOrString{Kind: IntstrInt, IntVal: val}
}

// NewIntOrStringFromString creates an IntOrString object with a string value.
func NewIntOrStringFromString(val string) IntOrString {
	return IntOrString{Kind: IntstrString, StrVal: val}
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (intstr *IntOrString) UnmarshalJSON(value []byte) error {
	if value[0] == '"' {
		intstr.Kind = IntstrString
		return json.Unmarshal(value, &intstr.StrVal)
	}
	intstr.Kind = IntstrInt
	return json.Unmarshal(value, &intstr.IntVal)
}

// String returns the string value, or Itoa's the int value.
func (intstr *IntOrString) String() string {
	if intstr.Kind == IntstrString {
		return intstr.StrVal
	}
	return strconv.Itoa(intstr.IntVal)
}

// MarshalJSON implements the json.Marshaller interface.
func (intstr IntOrString) MarshalJSON() ([]byte, error) {
	switch intstr.Kind {
	case IntstrInt:
		return json.Marshal(intstr.IntVal)
	case IntstrString:
		return json.Marshal(intstr.StrVal)
	default:
		return []byte{}, fmt.Errorf("impossible IntOrString.Kind")
	}
}

func (intstr *IntOrString) Fuzz(c fuzz.Continue) {
	if c.RandBool() {
		intstr.Kind = IntstrInt
		c.Fuzz(&intstr.IntVal)
		intstr.StrVal = ""
	} else {
		intstr.Kind = IntstrString
		intstr.IntVal = 0
		c.Fuzz(&intstr.StrVal)
	}
}

// Takes a list of strings and compiles them into a list of regular expressions
func CompileRegexps(regexpStrings []string) ([]*regexp.Regexp, error) {
	regexps := []*regexp.Regexp{}
	for _, regexpStr := range regexpStrings {
		r, err := regexp.Compile(regexpStr)
		if err != nil {
			return []*regexp.Regexp{}, err
		}
		regexps = append(regexps, r)
	}
	return regexps, nil
}

// Detects if using systemd as the init system
// Please note that simply reading /proc/1/cmdline can be misleading because
// some installation of various init programs can automatically make /sbin/init
// a symlink or even a renamed version of their main program.
// TODO(dchen1107): realiably detects the init system using on the system:
// systemd, upstart, initd, etc.
func UsingSystemdInitSystem() bool {
	if _, err := os.Stat("/run/systemd/system"); err == nil {
		return true
	}

	return false
}

// Tests whether all pointer fields in a struct are nil.  This is useful when,
// for example, an API struct is handled by plugins which need to distinguish
// "no plugin accepted this spec" from "this spec is empty".
//
// This function is only valid for structs and pointers to structs.  Any other
// type will cause a panic.  Passing a typed nil pointer will return true.
func AllPtrFieldsNil(obj interface{}) bool {
	v := reflect.ValueOf(obj)
	if !v.IsValid() {
		panic(fmt.Sprintf("reflect.ValueOf() produced a non-valid Value for %#v", obj))
	}
	if v.Kind() == reflect.Ptr {
		if v.IsNil() {
			return true
		}
		v = v.Elem()
	}
	for i := 0; i < v.NumField(); i++ {
		if v.Field(i).Kind() == reflect.Ptr && !v.Field(i).IsNil() {
			return false
		}
	}
	return true
}

// Splits a fully qualified name and returns its namespace and name.
// Assumes that the input 'str' has been validated.
func SplitQualifiedName(str string) (string, string) {
	parts := strings.Split(str, "/")
	if len(parts) < 2 {
		return "", str
	}

	return parts[0], parts[1]
}

// Joins 'namespace' and 'name' and returns a fully qualified name
// Assumes that the input is valid.
func JoinQualifiedName(namespace, name string) string {
	return path.Join(namespace, name)
}

type Route struct {
	Interface   string
	Destination net.IP
	Gateway     net.IP
	// TODO: add more fields here if needed
}

func getRoutes(input io.Reader) ([]Route, error) {
	routes := []Route{}
	if input == nil {
		return nil, fmt.Errorf("input is nil")
	}
	scanner := bufio.NewReader(input)
	for {
		line, err := scanner.ReadString('\n')
		if err == io.EOF {
			break
		}
		//ignore the headers in the route info
		if strings.HasPrefix(line, "Iface") {
			continue
		}
		fields := strings.Fields(line)
		routes = append(routes, Route{})
		route := &routes[len(routes)-1]
		route.Interface = fields[0]
		ip, err := parseIP(fields[1])
		if err != nil {
			return nil, err
		}
		route.Destination = ip
		ip, err = parseIP(fields[2])
		if err != nil {
			return nil, err
		}
		route.Gateway = ip
	}
	return routes, nil
}

func parseIP(str string) (net.IP, error) {
	if str == "" {
		return nil, fmt.Errorf("input is nil")
	}
	bytes, err := hex.DecodeString(str)
	if err != nil {
		return nil, err
	}
	//TODO add ipv6 support
	if len(bytes) != net.IPv4len {
		return nil, fmt.Errorf("only IPv4 is supported")
	}
	bytes[0], bytes[1], bytes[2], bytes[3] = bytes[3], bytes[2], bytes[1], bytes[0]
	return net.IP(bytes), nil
}

func isInterfaceUp(intf *net.Interface) bool {
	if intf == nil {
		return false
	}
	if intf.Flags&net.FlagUp != 0 {
		glog.V(4).Infof("Interface %v is up", intf.Name)
		return true
	}
	return false
}

//getFinalIP method receives all the IP addrs of a Interface
//and returns a nil if the address is Loopback , Ipv6  or nil.
//It returns a valid IPv4 if an Ipv4 address is found in the array.
func getFinalIP(addrs []net.Addr) (net.IP, error) {
	if len(addrs) > 0 {
		for i := range addrs {
			glog.V(4).Infof("Checking addr  %s.", addrs[i].String())
			ip, _, err := net.ParseCIDR(addrs[i].String())
			if err != nil {
				return nil, err
			}
			//Only IPv4
			//TODO : add IPv6 support
			if ip.To4() != nil {
				if !ip.IsLoopback() {
					glog.V(4).Infof("IP found %v", ip)
					return ip, nil
				} else {
					glog.V(4).Infof("Loopback found %v", ip)
				}
			} else {
				glog.V(4).Infof("%v is not a valid IPv4 address", ip)
			}

		}
	}
	return nil, nil
}

func getIPFromInterface(intfName string, nw networkInterfacer) (net.IP, error) {
	intf, err := nw.InterfaceByName(intfName)
	if err != nil {
		return nil, err
	}
	if isInterfaceUp(intf) {
		addrs, err := nw.Addrs(intf)
		if err != nil {
			return nil, err
		}
		glog.V(4).Infof("Interface %q has %d addresses :%v.", intfName, len(addrs), addrs)
		finalIP, err := getFinalIP(addrs)
		if err != nil {
			return nil, err
		}
		if finalIP != nil {
			glog.V(4).Infof("valid IPv4 address for interface %q found as %v.", intfName, finalIP)
			return finalIP, nil
		}
	}

	return nil, nil
}

func flagsSet(flags net.Flags, test net.Flags) bool {
	return flags&test != 0
}

func flagsClear(flags net.Flags, test net.Flags) bool {
	return flags&test == 0
}

func chooseHostInterfaceNativeGo() (net.IP, error) {
	intfs, err := net.Interfaces()
	if err != nil {
		return nil, err
	}
	i := 0
	var ip net.IP
	for i = range intfs {
		if flagsSet(intfs[i].Flags, net.FlagUp) && flagsClear(intfs[i].Flags, net.FlagLoopback|net.FlagPointToPoint) {
			addrs, err := intfs[i].Addrs()
			if err != nil {
				return nil, err
			}
			if len(addrs) > 0 {
				for _, addr := range addrs {
					if addrIP, _, err := net.ParseCIDR(addr.String()); err == nil {
						if addrIP.To4() != nil {
							ip = addrIP.To4()
							break
						}
					}
				}
				if ip != nil {
					// This interface should suffice.
					break
				}
			}
		}
	}
	if ip == nil {
		return nil, fmt.Errorf("no acceptable interface from host")
	}
	glog.V(4).Infof("Choosing interface %s (IP %v) as default", intfs[i].Name, ip)
	return ip, nil
}

//ChooseHostInterface is a method used fetch an IP for a daemon.
//It uses data from /proc/net/route file.
//For a node with no internet connection ,it returns error
//For a multi n/w interface node it returns the IP of the interface with gateway on it.
func ChooseHostInterface() (net.IP, error) {
	inFile, err := os.Open("/proc/net/route")
	if err != nil {
		if os.IsNotExist(err) {
			return chooseHostInterfaceNativeGo()
		}
		return nil, err
	}
	defer inFile.Close()
	var nw networkInterfacer = networkInterface{}
	return chooseHostInterfaceFromRoute(inFile, nw)
}

type networkInterfacer interface {
	InterfaceByName(intfName string) (*net.Interface, error)
	Addrs(intf *net.Interface) ([]net.Addr, error)
}

type networkInterface struct{}

func (_ networkInterface) InterfaceByName(intfName string) (*net.Interface, error) {
	intf, err := net.InterfaceByName(intfName)
	if err != nil {
		return nil, err
	}
	return intf, nil
}

func (_ networkInterface) Addrs(intf *net.Interface) ([]net.Addr, error) {
	addrs, err := intf.Addrs()
	if err != nil {
		return nil, err
	}
	return addrs, nil
}

func chooseHostInterfaceFromRoute(inFile io.Reader, nw networkInterfacer) (net.IP, error) {
	routes, err := getRoutes(inFile)
	if err != nil {
		return nil, err
	}
	zero := net.IP{0, 0, 0, 0}
	var finalIP net.IP
	for i := range routes {
		//find interface with gateway
		if routes[i].Destination.Equal(zero) {
			glog.V(4).Infof("Default route transits interface %q", routes[i].Interface)
			finalIP, err := getIPFromInterface(routes[i].Interface, nw)
			if err != nil {
				return nil, err
			}
			if finalIP != nil {
				glog.V(4).Infof("Choosing IP %v ", finalIP)
				return finalIP, nil
			}
		}
	}
	glog.V(4).Infof("No valid IP found")
	if finalIP == nil {
		return nil, fmt.Errorf("Unable to select an IP.")
	}
	return nil, nil
}

func GetClient(req *http.Request) string {
	if userAgent, ok := req.Header["User-Agent"]; ok {
		if len(userAgent) > 0 {
			return userAgent[0]
		}
	}
	return "unknown"
}

func ShortenString(str string, n int) string {
	if len(str) <= n {
		return str
	} else {
		return str[:n]
	}
}

func FileExists(filename string) (bool, error) {
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return false, nil
	} else if err != nil {
		return false, err
	}
	return true, nil
}
