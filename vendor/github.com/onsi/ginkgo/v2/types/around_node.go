package types

import (
	"context"
)

type AroundNodeAllowedFuncs interface {
	~func(context.Context, func(context.Context)) | ~func(context.Context) context.Context | ~func()
}
type AroundNodeFunc func(ctx context.Context, body func(ctx context.Context))

func AroundNode[F AroundNodeAllowedFuncs](f F, cl CodeLocation) AroundNodeDecorator {
	if f == nil {
		panic("BuildAroundNode cannot be called with a nil function.")
	}
	var aroundNodeFunc func(context.Context, func(context.Context))
	switch x := any(f).(type) {
	case func(context.Context, func(context.Context)):
		aroundNodeFunc = x
	case func(context.Context) context.Context:
		aroundNodeFunc = func(ctx context.Context, body func(context.Context)) {
			ctx = x(ctx)
			body(ctx)
		}
	case func():
		aroundNodeFunc = func(ctx context.Context, body func(context.Context)) {
			x()
			body(ctx)
		}
	}

	return AroundNodeDecorator{
		Body:         aroundNodeFunc,
		CodeLocation: cl,
	}
}

type AroundNodeDecorator struct {
	Body         AroundNodeFunc
	CodeLocation CodeLocation
}

type AroundNodes []AroundNodeDecorator

func (an AroundNodes) Clone() AroundNodes {
	out := make(AroundNodes, len(an))
	copy(out, an)
	return out
}

func (an AroundNodes) Append(other ...AroundNodeDecorator) AroundNodes {
	out := make(AroundNodes, len(an)+len(other))
	copy(out, an)
	copy(out[len(an):], other)
	return out
}
