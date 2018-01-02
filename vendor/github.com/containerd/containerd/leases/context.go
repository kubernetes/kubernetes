package leases

import "context"

type leaseKey struct{}

// WithLease sets a given lease on the context
func WithLease(ctx context.Context, lid string) context.Context {
	ctx = context.WithValue(ctx, leaseKey{}, lid)

	// also store on the grpc headers so it gets picked up by any clients that
	// are using this.
	return withGRPCLeaseHeader(ctx, lid)
}

// Lease returns the lease from the context.
func Lease(ctx context.Context) (string, bool) {
	lid, ok := ctx.Value(leaseKey{}).(string)
	if !ok {
		return fromGRPCHeader(ctx)
	}

	return lid, ok
}
