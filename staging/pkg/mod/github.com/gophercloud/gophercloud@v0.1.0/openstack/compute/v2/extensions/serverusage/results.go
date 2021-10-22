package serverusage

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
)

// UsageExt represents OS-SRV-USG server response fields.
type UsageExt struct {
	LaunchedAt   time.Time `json:"-"`
	TerminatedAt time.Time `json:"-"`
}

// UnmarshalJSON helps to unmarshal UsageExt fields into needed values.
func (r *UsageExt) UnmarshalJSON(b []byte) error {
	type tmp UsageExt
	var s struct {
		tmp
		LaunchedAt   gophercloud.JSONRFC3339MilliNoZ `json:"OS-SRV-USG:launched_at"`
		TerminatedAt gophercloud.JSONRFC3339MilliNoZ `json:"OS-SRV-USG:terminated_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = UsageExt(s.tmp)

	r.LaunchedAt = time.Time(s.LaunchedAt)
	r.TerminatedAt = time.Time(s.TerminatedAt)

	return nil
}
