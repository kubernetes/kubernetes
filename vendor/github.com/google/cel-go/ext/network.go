// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ext

import (
	"fmt"
	"net/netip"
	"reflect"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

const (
	// Version1 is the initial version of the Network library, providing
	// parity with Kubernetes v1.30+ CEL network functions.
	Version1 uint32 = 1
)

// Network returns a cel.EnvOption to configure extended functions for network
// address parsing, inspection, and CIDR range manipulation.
//
// Note: This library defines global functions `ip`, `cidr`, `isIP`, `isCIDR`
// and `ip.isCanonical`. If you are currently using variables named `ip` or
// `cidr`, these functions will likely work as intended, however there is a
// chance for collision.
//
// The library closely mirrors the behavior of the Kubernetes CEL network
// libraries, treating IP addresses and CIDR ranges as opaque types. It parses
// IPs strictly: IPv4-mapped IPv6 addresses and IP zones are not allowed.
//
// This library includes a TypeAdapter that allows `netip.Addr` and
// `netip.Prefix` Go types to be passed directly into the CEL environment.
//
// # IP Addresses
//
// The `ip` function converts a string to an IP address (IPv4 or IPv6). If the
// string is not a valid IP, an error is returned. The `isIP` function checks
// if a string is a valid IP address without throwing an error.
//
//	ip(string) -> ip
//	isIP(string) -> bool
//
// Examples:
//
//	ip('127.0.0.1')
//	ip('::1')
//	isIP('1.2.3.4') // true
//	isIP('invalid') // false
//
// # CIDR Ranges
//
// The `cidr` function converts a string to a Classless Inter-Domain Routing
// (CIDR) range. If the string is not valid, an error is returned.
//
// The `isCIDR` function checks if a string is a valid CIDR notation. Note that
// `isCIDR` allows CIDR values with or without host bits (e.g., '10.0.0.1/8'
// or '10.0.0.0/8').
//
//	cidr(string) -> cidr
//	isCIDR(string) -> bool
//
// Examples:
//
//	cidr('192.168.0.0/24')
//	cidr('::1/128')
//	isCIDR('10.0.0.0/8') // true
//	isCIDR('10.0.0.1/8') // true
//
// # IP Inspection and Canonicalization
//
// IP objects support various inspection methods.
//
//	<ip>.family() -> int
//	<ip>.isLoopback() -> bool
//	<ip>.isGlobalUnicast() -> bool
//	<ip>.isLinkLocalMulticast() -> bool
//	<ip>.isLinkLocalUnicast() -> bool
//	<ip>.isUnspecified() -> bool
//
// The `ip.isCanonical` function takes a string and returns true if it matches
// the RFC 5952 canonical string representation of that address.
//
//	ip.isCanonical(string) -> bool
//
// Examples:
//
//	ip('127.0.0.1').family() == 4
//	ip('::1').family() == 6
//	ip('127.0.0.1').isLoopback() == true
//	ip.isCanonical('2001:db8::1') == true  // RFC 5952 format
//	ip.isCanonical('2001:DB8::1') == false // Uppercase is not canonical
//	ip.isCanonical('2001:db8:0:0:0:0:0:1') == false // Expanded is not canonical
//
// # CIDR Member Functions
//
// CIDR objects support containment checks and property extraction.
//
//	<cidr>.containsIP(ip|string) -> bool
//	<cidr>.containsCIDR(cidr|string) -> bool
//	<cidr>.ip() -> ip
//	<cidr>.isMask() -> bool
//	<cidr>.masked() -> cidr
//	<cidr>.prefixLength() -> int
//
// Examples:
//
//	cidr('10.0.0.0/8').containsIP(ip('10.0.0.1')) == true
//	cidr('10.0.0.0/8').containsIP('10.0.0.1') == true
//	cidr('10.0.0.0/8').containsCIDR('10.1.0.0/16') == true
//	cidr('192.168.1.5/24').ip() == ip('192.168.1.5')
//	cidr('192.168.1.0/24').isMask() == true
//	cidr('192.168.1.5/24').isMask() == false
//	cidr('192.168.1.5/24').masked() == cidr('192.168.1.0/24')
//	cidr('192.168.1.0/24').prefixLength() == 24
func Network(opts ...NetworkOption) cel.EnvOption {
	lib := &networkLib{version: Version1}
	for _, o := range opts {
		lib = o(lib)
	}
	return func(e *cel.Env) (*cel.Env, error) {
		// Install the library (Types and Functions)
		e, err := cel.Lib(lib)(e)
		if err != nil {
			return nil, err
		}

		// Install the Adapter (Wrapping the existing one)
		adapter := &networkAdapter{Adapter: e.CELTypeAdapter()}
		return cel.CustomTypeAdapter(adapter)(e)
	}
}

// NetworkOption declares a functional operator for configuring the Network library behavior.
type NetworkOption func(*networkLib) *networkLib

// NetworkVersion sets the version of the network library to an explicit version.
func NetworkVersion(version uint32) NetworkOption {
	return func(lib *networkLib) *networkLib {
		lib.version = version
		return lib
	}
}

const (
	// Function names matching the original Kubernetes implementation of this networking library.
	// isStrictCIDR and isInterfaceAddress are added to enable strict isCIDR parsing without breaking
	// functionality for existing users. Ctx: https://github.com/kubernetes/kubernetes/issues/134224
	cidrFunc             = "cidr"
	cidrToString         = "string"
	containsCIDRFunc     = "containsCIDR"
	containsIPFunc       = "containsIP"
	familyFunc           = "family"
	ipFunc               = "ip"
	ipToString           = "string"
	isCanonicalFunc      = "ip.isCanonical"
	isCIDRFunc           = "isCIDR"
	isGlobalUnicastFunc  = "isGlobalUnicast"
	isIPFunc             = "isIP"
	isLinkLocalMcastFunc = "isLinkLocalMulticast"
	isLinkLocalUcastFunc = "isLinkLocalUnicast"
	isLoopbackFunc       = "isLoopback"
	isMaskFunc           = "isMask"
	isUnspecifiedFunc    = "isUnspecified"
	maskedFunc           = "masked"
	prefixLengthFunc     = "prefixLength"
)

var (
	// Definitions for the Opaque Types
	IPType   = types.NewOpaqueType("net.IP")
	CIDRType = types.NewOpaqueType("net.CIDR")
)

type networkLib struct {
	version uint32
}

func (*networkLib) LibraryName() string {
	return "cel.lib.ext.network"
}

func (*networkLib) CompileOptions() []cel.EnvOption {
	return []cel.EnvOption{
		// 1. Register Types
		cel.Types(
			IPType,
			CIDRType,
		),

		// 2. Register Functions
		cel.Function(cidrFunc,
			// K8s Parity: Following the pattern, this is "string_to_cidr"
			cel.Overload("string_to_cidr", []*cel.Type{cel.StringType}, CIDRType,
				cel.UnaryBinding(netCIDRString)),
		),
		cel.Function(cidrToString,
			cel.Overload("cidr_to_string", []*cel.Type{CIDRType}, cel.StringType,
				cel.UnaryBinding(netCIDRToString)),
		),
		cel.Function(containsCIDRFunc,
			cel.MemberOverload("cidr_contains_cidr", []*cel.Type{CIDRType, CIDRType}, cel.BoolType,
				cel.BinaryBinding(netCIDRContainsCIDR)),
			cel.MemberOverload("cidr_contains_cidr_string", []*cel.Type{CIDRType, cel.StringType}, cel.BoolType,
				cel.BinaryBinding(netCIDRContainsCIDRString)),
		),
		cel.Function(containsIPFunc,
			cel.MemberOverload("cidr_contains_ip_ip", []*cel.Type{CIDRType, IPType}, cel.BoolType,
				cel.BinaryBinding(netCIDRContainsIP)),
			cel.MemberOverload("cidr_contains_ip_string", []*cel.Type{CIDRType, cel.StringType}, cel.BoolType,
				cel.BinaryBinding(netCIDRContainsIPString)),
		),
		cel.Function(familyFunc,
			cel.MemberOverload("ip_family", []*cel.Type{IPType}, cel.IntType,
				cel.UnaryBinding(netIPFamily)),
		),
		cel.Function(ipFunc,
			// K8s Parity: The global overload is named "string_to_ip"
			cel.Overload("string_to_ip", []*cel.Type{cel.StringType}, IPType,
				cel.UnaryBinding(netIPString)),
			// K8s Parity: The member overload is named "cidr_ip"
			cel.MemberOverload("cidr_ip", []*cel.Type{CIDRType}, IPType,
				cel.UnaryBinding(netCIDRIP)),
		),
		cel.Function(ipToString,
			cel.Overload("ip_to_string", []*cel.Type{IPType}, cel.StringType,
				cel.UnaryBinding(netIPToString)),
		),
		cel.Function(isCanonicalFunc,
			cel.Overload("ip_is_canonical", []*cel.Type{cel.StringType}, cel.BoolType,
				cel.UnaryBinding(netIPIsCanonical)),
		),
		cel.Function(isCIDRFunc,
			cel.Overload("is_cidr", []*cel.Type{cel.StringType}, cel.BoolType,
				cel.UnaryBinding(netIsCIDR)),
		),
		cel.Function(isGlobalUnicastFunc,
			cel.MemberOverload("ip_is_global_unicast", []*cel.Type{IPType}, cel.BoolType,
				cel.UnaryBinding(netIPIsGlobalUnicast)),
		),
		cel.Function(isIPFunc,
			cel.Overload("is_ip", []*cel.Type{cel.StringType}, cel.BoolType,
				cel.UnaryBinding(netIsIP)),
		),
		cel.Function(isLinkLocalMcastFunc,
			cel.MemberOverload("ip_is_link_local_multicast", []*cel.Type{IPType}, cel.BoolType,
				cel.UnaryBinding(netIPIsLinkLocalMulticast)),
		),
		cel.Function(isLinkLocalUcastFunc,
			cel.MemberOverload("ip_is_link_local_unicast", []*cel.Type{IPType}, cel.BoolType,
				cel.UnaryBinding(netIPIsLinkLocalUnicast)),
		),
		cel.Function(isLoopbackFunc,
			cel.MemberOverload("ip_is_loopback", []*cel.Type{IPType}, cel.BoolType,
				cel.UnaryBinding(netIPIsLoopback)),
		),
		cel.Function(isMaskFunc,
			cel.MemberOverload("cidr_is_mask", []*cel.Type{CIDRType}, cel.BoolType,
				cel.UnaryBinding(netCIDRIsMask)),
		),
		cel.Function(isUnspecifiedFunc,
			cel.MemberOverload("ip_is_unspecified", []*cel.Type{IPType}, cel.BoolType,
				cel.UnaryBinding(netIPIsUnspecified)),
		),
		cel.Function(maskedFunc,
			cel.MemberOverload("cidr_masked", []*cel.Type{CIDRType}, CIDRType,
				cel.UnaryBinding(netCIDRMasked)),
		),
		cel.Function(prefixLengthFunc,
			cel.MemberOverload("cidr_prefix_length", []*cel.Type{CIDRType}, cel.IntType,
				cel.UnaryBinding(netCIDRPrefixLength)),
		),
		cel.ASTValidators(
			networkFormatValidator{funcName: ipFunc, argNum: 0, check: checkIP},
			networkFormatValidator{funcName: cidrFunc, argNum: 0, check: checkCIDR},
		),
	}
}

func (*networkLib) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

// networkAdapter adapts netip types while preserving existing adapters.
type networkAdapter struct {
	types.Adapter
}

func (a *networkAdapter) NativeToValue(value any) ref.Val {
	switch v := value.(type) {
	case netip.Addr:
		return IP{Addr: v}
	case netip.Prefix:
		return CIDR{Prefix: v}
	}
	// Delegate to the wrapped adapter (e.g., Protobuf adapter)
	return a.Adapter.NativeToValue(value)
}

// --- Implementation Logic ---

func netCIDRContainsCIDR(lhs, rhs ref.Val) ref.Val {
	parent := lhs.(CIDR)
	child := rhs.(CIDR)
	return types.Bool(parent.Prefix.Overlaps(child.Prefix) && parent.Prefix.Bits() <= child.Prefix.Bits())
}

func netCIDRContainsCIDRString(lhs, rhs ref.Val) ref.Val {
	parent := lhs.(CIDR)
	s := rhs.(types.String)
	childPrefix, err := parseCIDR(string(s))
	if err != nil {
		return types.WrapErr(err)
	}
	return types.Bool(parent.Prefix.Overlaps(childPrefix) && parent.Prefix.Bits() <= childPrefix.Bits())
}

func netCIDRContainsIP(lhs, rhs ref.Val) ref.Val {
	cidr := lhs.(CIDR)
	ip := rhs.(IP)
	return types.Bool(cidr.Prefix.Contains(ip.Addr))
}

func netCIDRContainsIPString(lhs, rhs ref.Val) ref.Val {
	cidr := lhs.(CIDR)
	s := rhs.(types.String)
	addr, err := parseIPAddr(string(s))
	if err != nil {
		return types.WrapErr(err)
	}
	return types.Bool(cidr.Prefix.Contains(addr))
}

func netCIDRIP(val ref.Val) ref.Val {
	cidr := val.(CIDR)
	return IP{Addr: cidr.Prefix.Addr()}
}

func netCIDRMasked(val ref.Val) ref.Val {
	cidr := val.(CIDR)
	return CIDR{Prefix: cidr.Prefix.Masked()}
}

func netCIDRPrefixLength(val ref.Val) ref.Val {
	cidr := val.(CIDR)
	return types.Int(cidr.Prefix.Bits())
}

func netCIDRString(val ref.Val) ref.Val {
	s := val.(types.String)
	str := string(s)
	prefix, err := parseCIDR(str)
	if err != nil {
		return types.WrapErr(err)
	}
	return CIDR{Prefix: prefix}
}

func netCIDRToString(val ref.Val) ref.Val {
	cidr := val.(CIDR)
	return types.String(cidr.Prefix.String())
}

func netIPFamily(val ref.Val) ref.Val {
	ip := val.(IP)
	if ip.Addr.Is4() {
		return types.Int(4)
	}
	return types.Int(6)
}

func netIPIsCanonical(val ref.Val) ref.Val {
	s := val.(types.String)
	str := string(s)
	addr, err := parseIPAddr(str)
	if err != nil {
		return types.WrapErr(err)
	}
	return types.Bool(addr.String() == str)
}

func netIPIsGlobalUnicast(val ref.Val) ref.Val {
	ip := val.(IP)
	return types.Bool(ip.Addr.IsGlobalUnicast())
}

func netIPIsLinkLocalMulticast(val ref.Val) ref.Val {
	ip := val.(IP)
	return types.Bool(ip.Addr.IsLinkLocalMulticast())
}

func netIPIsLinkLocalUnicast(val ref.Val) ref.Val {
	ip := val.(IP)
	return types.Bool(ip.Addr.IsLinkLocalUnicast())
}

func netIPIsLoopback(val ref.Val) ref.Val {
	ip := val.(IP)
	return types.Bool(ip.Addr.IsLoopback())
}

func netIPIsUnspecified(val ref.Val) ref.Val {
	ip := val.(IP)
	return types.Bool(ip.Addr.IsUnspecified())
}

func netIPString(val ref.Val) ref.Val {
	s := val.(types.String)
	str := string(s)
	addr, err := parseIPAddr(str)
	if err != nil {
		return types.WrapErr(err)
	}
	return IP{Addr: addr}
}

func netIPToString(val ref.Val) ref.Val {
	ip := val.(IP)
	return types.String(ip.Addr.String())
}

func netIsCIDR(val ref.Val) ref.Val {
	s := val.(types.String)
	_, err := parseCIDR(string(s))
	return types.Bool(err == nil)
}

func netIsIP(val ref.Val) ref.Val {
	s := val.(types.String)
	_, err := parseIPAddr(string(s))
	return types.Bool(err == nil)
}

func netCIDRIsMask(val ref.Val) ref.Val {
	cidr := val.(CIDR)
	return types.Bool(cidr.Prefix.Addr() == cidr.Prefix.Masked().Addr())
}

func parseCIDR(raw string) (netip.Prefix, error) {
	prefix, err := netip.ParsePrefix(raw)
	if err != nil {
		return netip.Prefix{}, fmt.Errorf("CIDR %q parse error during conversion from string: %v", raw, err)
	}
	if prefix.Addr().Zone() != "" {
		return netip.Prefix{}, fmt.Errorf("CIDR %q with zone value is not allowed", raw)
	}
	if prefix.Addr().Is4In6() {
		return netip.Prefix{}, fmt.Errorf("IPv4-mapped IPv6 address %q is not allowed", raw)
	}
	return prefix, nil
}

func parseIPAddr(raw string) (netip.Addr, error) {
	addr, err := netip.ParseAddr(raw)
	if err != nil {
		return netip.Addr{}, fmt.Errorf("IP Address %q parse error during conversion from string: %v", raw, err)
	}
	if addr.Zone() != "" {
		return netip.Addr{}, fmt.Errorf("IP address %q with zone value is not allowed", raw)
	}
	if addr.Is4In6() {
		return netip.Addr{}, fmt.Errorf("IPv4-mapped IPv6 address %q is not allowed", raw)
	}
	return addr, nil
}

// --- Opaque Type Wrappers ---

type IP struct {
	netip.Addr
}

// ConvertToNative converts the IP value to a native Go type.
func (i IP) ConvertToNative(typeDesc reflect.Type) (any, error) {
	if typeDesc == reflect.TypeFor[netip.Addr]() {
		return i.Addr, nil
	}
	if typeDesc.Kind() == reflect.String {
		return i.Addr.String(), nil
	}
	return nil, fmt.Errorf("unsupported type conversion to '%v'", typeDesc)
}

// ConvertToType converts the IP value to a CEL type.
func (i IP) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case types.StringType:
		return types.String(i.Addr.String())
	case IPType:
		return i
	case types.TypeType:
		return IPType
	}
	return types.NewErr("type conversion error from '%s' to '%s'", IPType, typeValue)
}

// Equal returns true if this IP is equal to the other ref.Val.
func (i IP) Equal(other ref.Val) ref.Val {
	o, ok := other.(IP)
	if !ok {
		return types.False
	}
	return types.Bool(i.Addr == o.Addr)
}

// Type returns the CEL type of the IP.
func (i IP) Type() ref.Type {
	return IPType
}

// Value returns the raw Go value (netip.Addr) of the IP.
func (i IP) Value() any {
	return i.Addr
}

type CIDR struct {
	netip.Prefix
}

// ConvertToNative converts the CIDR value to a native Go type.
func (c CIDR) ConvertToNative(typeDesc reflect.Type) (any, error) {
	if typeDesc == reflect.TypeFor[netip.Prefix]() {
		return c.Prefix, nil
	}
	if typeDesc.Kind() == reflect.String {
		return c.Prefix.String(), nil
	}
	return nil, fmt.Errorf("unsupported type conversion to '%v'", typeDesc)
}

// ConvertToType converts the CIDR value to a CEL type.
func (c CIDR) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case types.StringType:
		return types.String(c.Prefix.String())
	case CIDRType:
		return c
	case types.TypeType:
		return CIDRType
	}
	return types.NewErr("type conversion error from '%s' to '%s'", CIDRType, typeValue)
}

// Equal returns true if this CIDR is equal to the other ref.Val.
func (c CIDR) Equal(other ref.Val) ref.Val {
	o, ok := other.(CIDR)
	if !ok {
		return types.False
	}
	return types.Bool(c.Prefix == o.Prefix)
}

// Type returns the CEL type of the CIDR.
func (c CIDR) Type() ref.Type {
	return CIDRType
}

// Value returns the raw Go value (netip.Prefix) of the CIDR.
func (c CIDR) Value() any {
	return c.Prefix
}

// --- Static Validators ---

type argChecker func(e *cel.Env, call, arg ast.Expr) error

type networkFormatValidator struct {
	funcName string
	argNum   int
	check    argChecker
}

func (v networkFormatValidator) Name() string {
	return fmt.Sprintf("cel.validator.network.%s", v.funcName)
}

func (v networkFormatValidator) Validate(e *cel.Env, _ cel.ValidatorConfig, a *ast.AST, iss *cel.Issues) {
	root := ast.NavigateAST(a)
	funcCalls := ast.MatchDescendants(root, ast.FunctionMatcher(v.funcName))
	for _, call := range funcCalls {
		callArgs := call.AsCall().Args()
		if len(callArgs) <= v.argNum {
			continue
		}
		litArg := callArgs[v.argNum]
		if litArg.Kind() != ast.LiteralKind {
			continue
		}
		if err := v.check(e, call, litArg); err != nil {
			iss.ReportErrorAtID(litArg.ID(), "invalid %s argument: %v", v.funcName, err)
		}
	}
}

func checkIP(e *cel.Env, call, arg ast.Expr) error {
	pattern := arg.AsLiteral().Value().(string)
	_, err := parseIPAddr(pattern)
	return err
}

func checkCIDR(e *cel.Env, call, arg ast.Expr) error {
	pattern := arg.AsLiteral().Value().(string)
	_, err := parseCIDR(pattern)
	return err
}
