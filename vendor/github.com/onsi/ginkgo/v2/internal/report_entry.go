package internal

import (
	"time"

	"github.com/onsi/ginkgo/v2/types"
)

type ReportEntry = types.ReportEntry

func NewReportEntry(name string, cl types.CodeLocation, args ...any) (ReportEntry, error) {
	out := ReportEntry{
		Visibility: types.ReportEntryVisibilityAlways,
		Name:       name,
		Location:   cl,
		Time:       time.Now(),
	}
	var didSetValue = false
	for _, arg := range args {
		switch x := arg.(type) {
		case types.ReportEntryVisibility:
			out.Visibility = x
		case types.CodeLocation:
			out.Location = x
		case Offset:
			out.Location = types.NewCodeLocation(2 + int(x))
		case time.Time:
			out.Time = x
		default:
			if didSetValue {
				return ReportEntry{}, types.GinkgoErrors.TooManyReportEntryValues(out.Location, arg)
			}
			out.Value = types.WrapEntryValue(arg)
			didSetValue = true
		}
	}

	return out, nil
}
