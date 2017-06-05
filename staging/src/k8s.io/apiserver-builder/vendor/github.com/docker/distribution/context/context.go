package context

import (
	"sync"

	"github.com/docker/distribution/uuid"
	"golang.org/x/net/context"
)

// Context is a copy of Context from the golang.org/x/net/context package.
type Context interface {
	context.Context
}

// instanceContext is a context that provides only an instance id. It is
// provided as the main background context.
type instanceContext struct {
	Context
	id   string    // id of context, logged as "instance.id"
	once sync.Once // once protect generation of the id
}

func (ic *instanceContext) Value(key interface{}) interface{} {
	if key == "instance.id" {
		ic.once.Do(func() {
			// We want to lazy initialize the UUID such that we don't
			// call a random generator from the package initialization
			// code. For various reasons random could not be available
			// https://github.com/docker/distribution/issues/782
			ic.id = uuid.Generate().String()
		})
		return ic.id
	}

	return ic.Context.Value(key)
}

var background = &instanceContext{
	Context: context.Background(),
}

// Background returns a non-nil, empty Context. The background context
// provides a single key, "instance.id" that is globally unique to the
// process.
func Background() Context {
	return background
}

// WithValue returns a copy of parent in which the value associated with key is
// val. Use context Values only for request-scoped data that transits processes
// and APIs, not for passing optional parameters to functions.
func WithValue(parent Context, key, val interface{}) Context {
	return context.WithValue(parent, key, val)
}

// stringMapContext is a simple context implementation that checks a map for a
// key, falling back to a parent if not present.
type stringMapContext struct {
	context.Context
	m map[string]interface{}
}

// WithValues returns a context that proxies lookups through a map. Only
// supports string keys.
func WithValues(ctx context.Context, m map[string]interface{}) context.Context {
	mo := make(map[string]interface{}, len(m)) // make our own copy.
	for k, v := range m {
		mo[k] = v
	}

	return stringMapContext{
		Context: ctx,
		m:       mo,
	}
}

func (smc stringMapContext) Value(key interface{}) interface{} {
	if ks, ok := key.(string); ok {
		if v, ok := smc.m[ks]; ok {
			return v
		}
	}

	return smc.Context.Value(key)
}
