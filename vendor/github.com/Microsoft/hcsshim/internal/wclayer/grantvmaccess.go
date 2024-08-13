package wclayer

import (
	"context"

	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/oc"
	"go.opencensus.io/trace"
)

// GrantVmAccess adds access to a file for a given VM
func GrantVmAccess(ctx context.Context, vmid string, filepath string) (err error) {
	title := "hcsshim::GrantVmAccess"
	ctx, span := trace.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(
		trace.StringAttribute("vm-id", vmid),
		trace.StringAttribute("path", filepath))

	err = grantVmAccess(vmid, filepath)
	if err != nil {
		return hcserror.New(err, title+" - failed", "")
	}
	return nil
}
