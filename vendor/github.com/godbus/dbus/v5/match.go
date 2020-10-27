package dbus

import (
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

// doesn't make sense to export this option because clients can only
// subscribe to messages with signal type.
func withMatchType(typ string) MatchOption {
	return WithMatchOption("type", typ)
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
