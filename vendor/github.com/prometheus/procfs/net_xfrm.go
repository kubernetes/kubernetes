// Copyright 2017 Prometheus Team
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package procfs

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// XfrmStat models the contents of /proc/net/xfrm_stat.
type XfrmStat struct {
	// All errors which are not matched by other
	XfrmInError int
	// No buffer is left
	XfrmInBufferError int
	// Header Error
	XfrmInHdrError int
	// No state found
	// i.e. either inbound SPI, address, or IPSEC protocol at SA is wrong
	XfrmInNoStates int
	// Transformation protocol specific error
	// e.g. SA Key is wrong
	XfrmInStateProtoError int
	// Transformation mode specific error
	XfrmInStateModeError int
	// Sequence error
	// e.g. sequence number is out of window
	XfrmInStateSeqError int
	// State is expired
	XfrmInStateExpired int
	// State has mismatch option
	// e.g. UDP encapsulation type is mismatched
	XfrmInStateMismatch int
	// State is invalid
	XfrmInStateInvalid int
	// No matching template for states
	// e.g. Inbound SAs are correct but SP rule is wrong
	XfrmInTmplMismatch int
	// No policy is found for states
	// e.g. Inbound SAs are correct but no SP is found
	XfrmInNoPols int
	// Policy discards
	XfrmInPolBlock int
	// Policy error
	XfrmInPolError int
	// All errors which are not matched by others
	XfrmOutError int
	// Bundle generation error
	XfrmOutBundleGenError int
	// Bundle check error
	XfrmOutBundleCheckError int
	// No state was found
	XfrmOutNoStates int
	// Transformation protocol specific error
	XfrmOutStateProtoError int
	// Transportation mode specific error
	XfrmOutStateModeError int
	// Sequence error
	// i.e sequence number overflow
	XfrmOutStateSeqError int
	// State is expired
	XfrmOutStateExpired int
	// Policy discads
	XfrmOutPolBlock int
	// Policy is dead
	XfrmOutPolDead int
	// Policy Error
	XfrmOutPolError int
	// Forward routing of a packet is not allowed
	XfrmFwdHdrError int
	// State is invalid, perhaps expired
	XfrmOutStateInvalid int
	// State hasn’t been fully acquired before use
	XfrmAcquireError int
}

// NewXfrmStat reads the xfrm_stat statistics.
func NewXfrmStat() (XfrmStat, error) {
	fs, err := NewFS(DefaultMountPoint)
	if err != nil {
		return XfrmStat{}, err
	}

	return fs.NewXfrmStat()
}

// NewXfrmStat reads the xfrm_stat statistics from the 'proc' filesystem.
func (fs FS) NewXfrmStat() (XfrmStat, error) {
	file, err := os.Open(fs.proc.Path("net/xfrm_stat"))
	if err != nil {
		return XfrmStat{}, err
	}
	defer file.Close()

	var (
		x = XfrmStat{}
		s = bufio.NewScanner(file)
	)

	for s.Scan() {
		fields := strings.Fields(s.Text())

		if len(fields) != 2 {
			return XfrmStat{}, fmt.Errorf("%w: %q line %q", ErrFileParse, file.Name(), s.Text())
		}

		name := fields[0]
		value, err := strconv.Atoi(fields[1])
		if err != nil {
			return XfrmStat{}, err
		}

		switch name {
		case "XfrmInError":
			x.XfrmInError = value
		case "XfrmInBufferError":
			x.XfrmInBufferError = value
		case "XfrmInHdrError":
			x.XfrmInHdrError = value
		case "XfrmInNoStates":
			x.XfrmInNoStates = value
		case "XfrmInStateProtoError":
			x.XfrmInStateProtoError = value
		case "XfrmInStateModeError":
			x.XfrmInStateModeError = value
		case "XfrmInStateSeqError":
			x.XfrmInStateSeqError = value
		case "XfrmInStateExpired":
			x.XfrmInStateExpired = value
		case "XfrmInStateInvalid":
			x.XfrmInStateInvalid = value
		case "XfrmInTmplMismatch":
			x.XfrmInTmplMismatch = value
		case "XfrmInNoPols":
			x.XfrmInNoPols = value
		case "XfrmInPolBlock":
			x.XfrmInPolBlock = value
		case "XfrmInPolError":
			x.XfrmInPolError = value
		case "XfrmOutError":
			x.XfrmOutError = value
		case "XfrmInStateMismatch":
			x.XfrmInStateMismatch = value
		case "XfrmOutBundleGenError":
			x.XfrmOutBundleGenError = value
		case "XfrmOutBundleCheckError":
			x.XfrmOutBundleCheckError = value
		case "XfrmOutNoStates":
			x.XfrmOutNoStates = value
		case "XfrmOutStateProtoError":
			x.XfrmOutStateProtoError = value
		case "XfrmOutStateModeError":
			x.XfrmOutStateModeError = value
		case "XfrmOutStateSeqError":
			x.XfrmOutStateSeqError = value
		case "XfrmOutStateExpired":
			x.XfrmOutStateExpired = value
		case "XfrmOutPolBlock":
			x.XfrmOutPolBlock = value
		case "XfrmOutPolDead":
			x.XfrmOutPolDead = value
		case "XfrmOutPolError":
			x.XfrmOutPolError = value
		case "XfrmFwdHdrError":
			x.XfrmFwdHdrError = value
		case "XfrmOutStateInvalid":
			x.XfrmOutStateInvalid = value
		case "XfrmAcquireError":
			x.XfrmAcquireError = value
		}

	}

	return x, s.Err()
}
