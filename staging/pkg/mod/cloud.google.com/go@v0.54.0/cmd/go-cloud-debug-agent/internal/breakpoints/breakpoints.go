// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package breakpoints handles breakpoint requests we get from the user through
// the Debuglet Controller, and manages corresponding breakpoints set in the code.
package breakpoints

import (
	"log"
	"sync"

	"cloud.google.com/go/cmd/go-cloud-debug-agent/internal/debug"
	cd "google.golang.org/api/clouddebugger/v2"
)

// BreakpointStore stores the set of breakpoints for a program.
type BreakpointStore struct {
	mu sync.Mutex
	// prog is the program being debugged.
	prog debug.Program
	// idToBreakpoint is a map from breakpoint identifier to *cd.Breakpoint.  The
	// map value is nil if the breakpoint is inactive.  A breakpoint is active if:
	// - We received it from the Debuglet Controller, and it was active at the time;
	// - We were able to set code breakpoints for it;
	// - We have not reached any of those code breakpoints while satisfying the
	//   breakpoint's conditions, or the breakpoint has action LOG; and
	// - The Debuglet Controller hasn't informed us the breakpoint has become inactive.
	idToBreakpoint map[string]*cd.Breakpoint
	// pcToBps and bpToPCs store the many-to-many relationship between breakpoints we
	// received from the Debuglet Controller and the code breakpoints we set for them.
	pcToBps map[uint64][]*cd.Breakpoint
	bpToPCs map[*cd.Breakpoint][]uint64
	// errors contains any breakpoints which couldn't be set because they caused an
	// error.  These are retrieved with ErrorBreakpoints, and the caller is
	// expected to handle sending updates for them.
	errors []*cd.Breakpoint
}

// NewBreakpointStore returns a BreakpointStore for the given program.
func NewBreakpointStore(prog debug.Program) *BreakpointStore {
	return &BreakpointStore{
		idToBreakpoint: make(map[string]*cd.Breakpoint),
		pcToBps:        make(map[uint64][]*cd.Breakpoint),
		bpToPCs:        make(map[*cd.Breakpoint][]uint64),
		prog:           prog,
	}
}

// ProcessBreakpointList applies updates received from the Debuglet Controller through a List call.
func (bs *BreakpointStore) ProcessBreakpointList(bps []*cd.Breakpoint) {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	for _, bp := range bps {
		if storedBp, ok := bs.idToBreakpoint[bp.Id]; ok {
			if storedBp != nil && bp.IsFinalState {
				// IsFinalState indicates that the breakpoint has been made inactive.
				bs.removeBreakpointLocked(storedBp)
			}
		} else {
			if bp.IsFinalState {
				// The controller is notifying us that the breakpoint is no longer active,
				// but we didn't know about it anyway.
				continue
			}
			if bp.Action != "" && bp.Action != "CAPTURE" && bp.Action != "LOG" {
				bp.IsFinalState = true
				bp.Status = &cd.StatusMessage{
					Description: &cd.FormatMessage{Format: "Action is not supported"},
					IsError:     true,
				}
				bs.errors = append(bs.errors, bp)
				// Note in idToBreakpoint that we've already seen this breakpoint, so that we
				// don't try to report it as an error multiple times.
				bs.idToBreakpoint[bp.Id] = nil
				continue
			}
			pcs, err := bs.prog.BreakpointAtLine(bp.Location.Path, uint64(bp.Location.Line))
			if err != nil {
				log.Printf("error setting breakpoint at %s:%d: %v", bp.Location.Path, bp.Location.Line, err)
			}
			if len(pcs) == 0 {
				// We can't find a PC for this breakpoint's source line, so don't make it active.
				// TODO: we could snap the line to a location where we can break, or report an error to the user.
				bs.idToBreakpoint[bp.Id] = nil
			} else {
				bs.idToBreakpoint[bp.Id] = bp
				for _, pc := range pcs {
					bs.pcToBps[pc] = append(bs.pcToBps[pc], bp)
				}
				bs.bpToPCs[bp] = pcs
			}
		}
	}
}

// ErrorBreakpoints returns a slice of Breakpoints that caused errors when the
// BreakpointStore tried to process them, and resets the list of such
// breakpoints.
// The caller is expected to send updates to the server to indicate the errors.
func (bs *BreakpointStore) ErrorBreakpoints() []*cd.Breakpoint {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	bps := bs.errors
	bs.errors = nil
	return bps
}

// BreakpointsAtPC returns all the breakpoints for which we set a code
// breakpoint at the given address.
func (bs *BreakpointStore) BreakpointsAtPC(pc uint64) []*cd.Breakpoint {
	bs.mu.Lock()
	defer bs.mu.Unlock()
	return bs.pcToBps[pc]
}

// RemoveBreakpoint makes the given breakpoint inactive.
// This is called when either the debugged program hits the breakpoint, or the Debuglet
// Controller informs us that the breakpoint is now inactive.
func (bs *BreakpointStore) RemoveBreakpoint(bp *cd.Breakpoint) {
	bs.mu.Lock()
	bs.removeBreakpointLocked(bp)
	bs.mu.Unlock()
}

func (bs *BreakpointStore) removeBreakpointLocked(bp *cd.Breakpoint) {
	// Set the ID's corresponding breakpoint to nil, so that we won't activate it
	// if we see it again.
	// TODO: we could delete it after a few seconds.
	bs.idToBreakpoint[bp.Id] = nil

	// Delete bp from the list of cd breakpoints at each of its corresponding
	// code breakpoint locations, and delete any code breakpoints which no longer
	// have a corresponding cd breakpoint.
	var codeBreakpointsToDelete []uint64
	for _, pc := range bs.bpToPCs[bp] {
		bps := remove(bs.pcToBps[pc], bp)
		if len(bps) == 0 {
			// bp was the last breakpoint set at this PC, so delete the code breakpoint.
			codeBreakpointsToDelete = append(codeBreakpointsToDelete, pc)
			delete(bs.pcToBps, pc)
		} else {
			bs.pcToBps[pc] = bps
		}
	}
	if len(codeBreakpointsToDelete) > 0 {
		bs.prog.DeleteBreakpoints(codeBreakpointsToDelete)
	}
	delete(bs.bpToPCs, bp)
}

// remove updates rs by removing r, then returns rs.
// The mutex in the BreakpointStore which contains rs should be held.
func remove(rs []*cd.Breakpoint, r *cd.Breakpoint) []*cd.Breakpoint {
	for i := range rs {
		if rs[i] == r {
			rs[i] = rs[len(rs)-1]
			rs = rs[0 : len(rs)-1]
			return rs
		}
	}
	// We shouldn't reach here.
	return rs
}
