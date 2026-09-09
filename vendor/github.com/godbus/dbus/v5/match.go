package dbus

import (
	"strconv"
	"strings"
)

// MatchOption specifies option for dbus routing match rule. Options can be constructed with WithMatch* helpers.
// For full list of available options consult
// https://dbus.freedesktop.org/doc/dbus-specification.html#message-bus-routing-match-rules
type MatchOption struct {
	key   string
	value string
}

func formatMatchOptions(options []MatchOption) string {
	items := make([]string, 0, len(options))
	for _, option := range options {
		items = append(items, option.key+"='"+option.value+"'")
	}
	return strings.Join(items, ",")
}

// WithMatchOption creates match option with given key and value
func WithMatchOption(key, value string) MatchOption {
	return MatchOption{key, value}
}

// It does not make sense to have a public WithMatchType function
// because clients can only subscribe to messages with signal type.
func withMatchTypeSignal() MatchOption {
	return WithMatchOption("type", "signal")
}

// WithMatchSender sets sender match option.
func WithMatchSender(sender string) MatchOption {
	return WithMatchOption("sender", sender)
}

// WithMatchSender sets interface match option.
func WithMatchInterface(iface string) MatchOption {
	return WithMatchOption("interface", iface)
}

// WithMatchMember sets member match option.
func WithMatchMember(member string) MatchOption {
	return WithMatchOption("member", member)
}

// WithMatchObjectPath creates match option that filters events based on given path
func WithMatchObjectPath(path ObjectPath) MatchOption {
	return WithMatchOption("path", string(path))
}

// WithMatchPathNamespace sets path_namespace match option.
func WithMatchPathNamespace(namespace ObjectPath) MatchOption {
	return WithMatchOption("path_namespace", string(namespace))
}

// WithMatchDestination sets destination match option.
func WithMatchDestination(destination string) MatchOption {
	return WithMatchOption("destination", destination)
}

// WithMatchArg sets argN match option, range of N is 0 to 63.
func WithMatchArg(argIdx int, value string) MatchOption {
	if argIdx < 0 || argIdx > 63 {
		panic("range of argument index is 0 to 63")
	}
	return WithMatchOption("arg"+strconv.Itoa(argIdx), value)
}

// WithMatchArgPath sets argN path match option, range of N is 0 to 63.
func WithMatchArgPath(argIdx int, path string) MatchOption {
	if argIdx < 0 || argIdx > 63 {
		panic("range of argument index is 0 to 63")
	}
	return WithMatchOption("arg"+strconv.Itoa(argIdx)+"path", path)
}

// WithMatchArg0Namespace sets arg0namespace match option.
func WithMatchArg0Namespace(arg0Namespace string) MatchOption {
	return WithMatchOption("arg0namespace", arg0Namespace)
}

// WithMatchEavesdrop sets eavesdrop match option.
func WithMatchEavesdrop(eavesdrop bool) MatchOption {
	return WithMatchOption("eavesdrop", strconv.FormatBool(eavesdrop))
}
