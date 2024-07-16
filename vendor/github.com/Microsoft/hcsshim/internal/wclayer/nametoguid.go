//go:build windows

package wclayer

import (
	"context"

	"github.com/Microsoft/go-winio/pkg/guid"
	"github.com/Microsoft/hcsshim/internal/hcserror"
	"github.com/Microsoft/hcsshim/internal/oc"
	"go.opencensus.io/trace"
)

// NameToGuid converts the given string into a GUID using the algorithm in the
// Host Compute Service, ensuring GUIDs generated with the same string are common
// across all clients.
func NameToGuid(ctx context.Context, name string) (_ guid.GUID, err error) {
	title := "hcsshim::NameToGuid"
	ctx, span := oc.StartSpan(ctx, title) //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(trace.StringAttribute("objectName", name))

	var id guid.GUID
	err = nameToGuid(name, &id)
	if err != nil {
		return guid.GUID{}, hcserror.New(err, title, "")
	}
	span.AddAttributes(trace.StringAttribute("guid", id.String()))
	return id, nil
}
