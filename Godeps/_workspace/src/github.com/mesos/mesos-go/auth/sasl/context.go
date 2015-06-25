package sasl

import (
	"net"

	"golang.org/x/net/context"
)

// unexported to prevent collisions with context keys defined in
// other packages.
type _key int

// If this package defined other context keys, they would have
// different integer values.
const (
	statusKey         _key = iota
	bindingAddressKey      // bind address for login-related network ops
)

func withStatus(ctx context.Context, s statusType) context.Context {
	return context.WithValue(ctx, statusKey, s)
}

func statusFrom(ctx context.Context) statusType {
	s, ok := ctx.Value(statusKey).(statusType)
	if !ok {
		panic("missing status in context")
	}
	return s
}

func WithBindingAddress(ctx context.Context, address net.IP) context.Context {
	return context.WithValue(ctx, bindingAddressKey, address)
}

func BindingAddressFrom(ctx context.Context) net.IP {
	obj := ctx.Value(bindingAddressKey)
	if addr, ok := obj.(net.IP); ok {
		return addr
	} else {
		return nil
	}
}
