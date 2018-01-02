package leases

import (
	"golang.org/x/net/context"
	"google.golang.org/grpc/metadata"
)

const (
	// GRPCHeader defines the header name for specifying a containerd lease.
	GRPCHeader = "containerd-lease"
)

func withGRPCLeaseHeader(ctx context.Context, lid string) context.Context {
	// also store on the grpc headers so it gets picked up by any clients
	// that are using this.
	txheader := metadata.Pairs(GRPCHeader, lid)
	md, ok := metadata.FromOutgoingContext(ctx) // merge with outgoing context.
	if !ok {
		md = txheader
	} else {
		// order ensures the latest is first in this list.
		md = metadata.Join(txheader, md)
	}

	return metadata.NewOutgoingContext(ctx, md)
}

func fromGRPCHeader(ctx context.Context) (string, bool) {
	// try to extract for use in grpc servers.
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return "", false
	}

	values := md[GRPCHeader]
	if len(values) == 0 {
		return "", false
	}

	return values[0], true
}
