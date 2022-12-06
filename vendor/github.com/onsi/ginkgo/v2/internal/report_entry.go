package internal

import (
	"reflect"
	"time"

	"github.com/onsi/ginkgo/v2/types"
)

type ReportEntry = types.ReportEntry

func NewReportEntry(name string, cl types.CodeLocation, args ...interface{}) (ReportEntry, error) {
	out := ReportEntry{
		Visibility: types.ReportEntryVisibilityAlways,
		Name:       name,
		Time:       time.Now(),
		Location:   cl,
	}
	var didSetValue = false
	for _, arg := range args {
		switch reflect.TypeOf(arg) {
		case reflect.TypeOf(types.ReportEntryVisibilityAlways):
			out.Visibility = arg.(types.ReportEntryVisibility)
		case reflect.TypeOf(types.CodeLocation{}):
			out.Location = arg.(types.CodeLocation)
		case reflect.TypeOf(Offset(0)):
			out.Location = types.NewCodeLocation(2 + int(arg.(Offset)))
		case reflect.TypeOf(out.Time):
			out.Time = arg.(time.Time)
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
