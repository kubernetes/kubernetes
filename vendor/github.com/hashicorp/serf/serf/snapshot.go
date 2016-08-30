package serf

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/armon/go-metrics"
	"github.com/hashicorp/serf/coordinate"
)

/*
Serf supports using a "snapshot" file that contains various
transactional data that is used to help Serf recover quickly
and gracefully from a failure. We append member events, as well
as the latest clock values to the file during normal operation,
and periodically checkpoint and roll over the file. During a restore,
we can replay the various member events to recall a list of known
nodes to re-join, as well as restore our clock values to avoid replaying
old events.
*/

const flushInterval = 500 * time.Millisecond
const clockUpdateInterval = 500 * time.Millisecond
const coordinateUpdateInterval = 60 * time.Second
const tmpExt = ".compact"

// Snapshotter is responsible for ingesting events and persisting
// them to disk, and providing a recovery mechanism at start time.
type Snapshotter struct {
	aliveNodes       map[string]string
	clock            *LamportClock
	coordClient      *coordinate.Client
	fh               *os.File
	buffered         *bufio.Writer
	inCh             <-chan Event
	lastFlush        time.Time
	lastClock        LamportTime
	lastEventClock   LamportTime
	lastQueryClock   LamportTime
	leaveCh          chan struct{}
	leaving          bool
	logger           *log.Logger
	maxSize          int64
	path             string
	offset           int64
	outCh            chan<- Event
	rejoinAfterLeave bool
	shutdownCh       <-chan struct{}
	waitCh           chan struct{}
}

// PreviousNode is used to represent the previously known alive nodes
type PreviousNode struct {
	Name string
	Addr string
}

func (p PreviousNode) String() string {
	return fmt.Sprintf("%s: %s", p.Name, p.Addr)
}

// NewSnapshotter creates a new Snapshotter that records events up to a
// max byte size before rotating the file. It can also be used to
// recover old state. Snapshotter works by reading an event channel it returns,
// passing through to an output channel, and persisting relevant events to disk.
// Setting rejoinAfterLeave makes leave not clear the state, and can be used
// if you intend to rejoin the same cluster after a leave.
func NewSnapshotter(path string,
	maxSize int,
	rejoinAfterLeave bool,
	logger *log.Logger,
	clock *LamportClock,
	coordClient *coordinate.Client,
	outCh chan<- Event,
	shutdownCh <-chan struct{}) (chan<- Event, *Snapshotter, error) {
	inCh := make(chan Event, 1024)

	// Try to open the file
	fh, err := os.OpenFile(path, os.O_RDWR|os.O_APPEND|os.O_CREATE, 0755)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open snapshot: %v", err)
	}

	// Determine the offset
	info, err := fh.Stat()
	if err != nil {
		fh.Close()
		return nil, nil, fmt.Errorf("failed to stat snapshot: %v", err)
	}
	offset := info.Size()

	// Create the snapshotter
	snap := &Snapshotter{
		aliveNodes:       make(map[string]string),
		clock:            clock,
		coordClient:      coordClient,
		fh:               fh,
		buffered:         bufio.NewWriter(fh),
		inCh:             inCh,
		lastClock:        0,
		lastEventClock:   0,
		lastQueryClock:   0,
		leaveCh:          make(chan struct{}),
		logger:           logger,
		maxSize:          int64(maxSize),
		path:             path,
		offset:           offset,
		outCh:            outCh,
		rejoinAfterLeave: rejoinAfterLeave,
		shutdownCh:       shutdownCh,
		waitCh:           make(chan struct{}),
	}

	// Recover the last known state
	if err := snap.replay(); err != nil {
		fh.Close()
		return nil, nil, err
	}

	// Start handling new commands
	go snap.stream()
	return inCh, snap, nil
}

// LastClock returns the last known clock time
func (s *Snapshotter) LastClock() LamportTime {
	return s.lastClock
}

// LastEventClock returns the last known event clock time
func (s *Snapshotter) LastEventClock() LamportTime {
	return s.lastEventClock
}

// LastQueryClock returns the last known query clock time
func (s *Snapshotter) LastQueryClock() LamportTime {
	return s.lastQueryClock
}

// AliveNodes returns the last known alive nodes
func (s *Snapshotter) AliveNodes() []*PreviousNode {
	// Copy the previously known
	previous := make([]*PreviousNode, 0, len(s.aliveNodes))
	for name, addr := range s.aliveNodes {
		previous = append(previous, &PreviousNode{name, addr})
	}

	// Randomize the order, prevents hot shards
	for i := range previous {
		j := rand.Intn(i + 1)
		previous[i], previous[j] = previous[j], previous[i]
	}
	return previous
}

// Wait is used to wait until the snapshotter finishes shut down
func (s *Snapshotter) Wait() {
	<-s.waitCh
}

// Leave is used to remove known nodes to prevent a restart from
// causing a join. Otherwise nodes will re-join after leaving!
func (s *Snapshotter) Leave() {
	select {
	case s.leaveCh <- struct{}{}:
	case <-s.shutdownCh:
	}
}

// stream is a long running routine that is used to handle events
func (s *Snapshotter) stream() {
	clockTicker := time.NewTicker(clockUpdateInterval)
	defer clockTicker.Stop()

	coordinateTicker := time.NewTicker(coordinateUpdateInterval)
	defer coordinateTicker.Stop()

	for {
		select {
		case <-s.leaveCh:
			s.leaving = true

			// If we plan to re-join, keep our state
			if !s.rejoinAfterLeave {
				s.aliveNodes = make(map[string]string)
			}
			s.tryAppend("leave\n")
			if err := s.buffered.Flush(); err != nil {
				s.logger.Printf("[ERR] serf: failed to flush leave to snapshot: %v", err)
			}
			if err := s.fh.Sync(); err != nil {
				s.logger.Printf("[ERR] serf: failed to sync leave to snapshot: %v", err)
			}

		case e := <-s.inCh:
			// Forward the event immediately
			if s.outCh != nil {
				s.outCh <- e
			}

			// Stop recording events after a leave is issued
			if s.leaving {
				continue
			}
			switch typed := e.(type) {
			case MemberEvent:
				s.processMemberEvent(typed)
			case UserEvent:
				s.processUserEvent(typed)
			case *Query:
				s.processQuery(typed)
			default:
				s.logger.Printf("[ERR] serf: Unknown event to snapshot: %#v", e)
			}

		case <-clockTicker.C:
			s.updateClock()

		case <-coordinateTicker.C:
			s.updateCoordinate()

		case <-s.shutdownCh:
			if err := s.buffered.Flush(); err != nil {
				s.logger.Printf("[ERR] serf: failed to flush snapshot: %v", err)
			}
			if err := s.fh.Sync(); err != nil {
				s.logger.Printf("[ERR] serf: failed to sync snapshot: %v", err)
			}
			s.fh.Close()
			close(s.waitCh)
			return
		}
	}
}

// processMemberEvent is used to handle a single member event
func (s *Snapshotter) processMemberEvent(e MemberEvent) {
	switch e.Type {
	case EventMemberJoin:
		for _, mem := range e.Members {
			addr := net.TCPAddr{IP: mem.Addr, Port: int(mem.Port)}
			s.aliveNodes[mem.Name] = addr.String()
			s.tryAppend(fmt.Sprintf("alive: %s %s\n", mem.Name, addr.String()))
		}

	case EventMemberLeave:
		fallthrough
	case EventMemberFailed:
		for _, mem := range e.Members {
			delete(s.aliveNodes, mem.Name)
			s.tryAppend(fmt.Sprintf("not-alive: %s\n", mem.Name))
		}
	}
	s.updateClock()
}

// updateClock is called periodically to check if we should udpate our
// clock value. This is done after member events but should also be done
// periodically due to race conditions with join and leave intents
func (s *Snapshotter) updateClock() {
	lastSeen := s.clock.Time() - 1
	if lastSeen > s.lastClock {
		s.lastClock = lastSeen
		s.tryAppend(fmt.Sprintf("clock: %d\n", s.lastClock))
	}
}

// updateCoordinate is called periodically to write out the current local
// coordinate. It's safe to call this if coordinates aren't enabled (nil
// client) and it will be a no-op.
func (s *Snapshotter) updateCoordinate() {
	if s.coordClient != nil {
		encoded, err := json.Marshal(s.coordClient.GetCoordinate())
		if err != nil {
			s.logger.Printf("[ERR] serf: Failed to encode coordinate: %v", err)
		} else {
			s.tryAppend(fmt.Sprintf("coordinate: %s\n", encoded))
		}
	}
}

// processUserEvent is used to handle a single user event
func (s *Snapshotter) processUserEvent(e UserEvent) {
	// Ignore old clocks
	if e.LTime <= s.lastEventClock {
		return
	}
	s.lastEventClock = e.LTime
	s.tryAppend(fmt.Sprintf("event-clock: %d\n", e.LTime))
}

// processQuery is used to handle a single query event
func (s *Snapshotter) processQuery(q *Query) {
	// Ignore old clocks
	if q.LTime <= s.lastQueryClock {
		return
	}
	s.lastQueryClock = q.LTime
	s.tryAppend(fmt.Sprintf("query-clock: %d\n", q.LTime))
}

// tryAppend will invoke append line but will not return an error
func (s *Snapshotter) tryAppend(l string) {
	if err := s.appendLine(l); err != nil {
		s.logger.Printf("[ERR] serf: Failed to update snapshot: %v", err)
	}
}

// appendLine is used to append a line to the existing log
func (s *Snapshotter) appendLine(l string) error {
	defer metrics.MeasureSince([]string{"serf", "snapshot", "appendLine"}, time.Now())

	n, err := s.buffered.WriteString(l)
	if err != nil {
		return err
	}

	// Check if we should flush
	now := time.Now()
	if now.Sub(s.lastFlush) > flushInterval {
		s.lastFlush = now
		if err := s.buffered.Flush(); err != nil {
			return err
		}
	}

	// Check if a compaction is necessary
	s.offset += int64(n)
	if s.offset > s.maxSize {
		return s.compact()
	}
	return nil
}

// Compact is used to compact the snapshot once it is too large
func (s *Snapshotter) compact() error {
	defer metrics.MeasureSince([]string{"serf", "snapshot", "compact"}, time.Now())

	// Try to open the file to new fiel
	newPath := s.path + tmpExt
	fh, err := os.OpenFile(newPath, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0755)
	if err != nil {
		return fmt.Errorf("failed to open new snapshot: %v", err)
	}

	// Create a buffered writer
	buf := bufio.NewWriter(fh)

	// Write out the live nodes
	var offset int64
	for name, addr := range s.aliveNodes {
		line := fmt.Sprintf("alive: %s %s\n", name, addr)
		n, err := buf.WriteString(line)
		if err != nil {
			fh.Close()
			return err
		}
		offset += int64(n)
	}

	// Write out the clocks
	line := fmt.Sprintf("clock: %d\n", s.lastClock)
	n, err := buf.WriteString(line)
	if err != nil {
		fh.Close()
		return err
	}
	offset += int64(n)

	line = fmt.Sprintf("event-clock: %d\n", s.lastEventClock)
	n, err = buf.WriteString(line)
	if err != nil {
		fh.Close()
		return err
	}
	offset += int64(n)

	line = fmt.Sprintf("query-clock: %d\n", s.lastQueryClock)
	n, err = buf.WriteString(line)
	if err != nil {
		fh.Close()
		return err
	}
	offset += int64(n)

	// Write out the coordinate.
	if s.coordClient != nil {
		encoded, err := json.Marshal(s.coordClient.GetCoordinate())
		if err != nil {
			fh.Close()
			return err
		}

		line = fmt.Sprintf("coordinate: %s\n", encoded)
		n, err = buf.WriteString(line)
		if err != nil {
			fh.Close()
			return err
		}
		offset += int64(n)
	}

	// Flush the new snapshot
	err = buf.Flush()
	fh.Close()
	if err != nil {
		return fmt.Errorf("failed to flush new snapshot: %v", err)
	}

	// We now need to swap the old snapshot file with the new snapshot.
	// Turns out, Windows won't let us rename the files if we have
	// open handles to them or if the destination already exists. This
	// means we are forced to close the existing handles, delete the
	// old file, move the new one in place, and then re-open the file
	// handles.

	// Flush the existing snapshot, ignoring errors since we will
	// delete it momentarily.
	s.buffered.Flush()
	s.buffered = nil

	// Close the file handle to the old snapshot
	s.fh.Close()
	s.fh = nil

	// Delete the old file
	if err := os.Remove(s.path); err != nil {
		return fmt.Errorf("failed to remove old snapshot: %v", err)
	}

	// Move the new file into place
	if err := os.Rename(newPath, s.path); err != nil {
		return fmt.Errorf("failed to install new snapshot: %v", err)
	}

	// Open the new snapshot
	fh, err = os.OpenFile(s.path, os.O_RDWR|os.O_APPEND|os.O_CREATE, 0755)
	if err != nil {
		return fmt.Errorf("failed to open snapshot: %v", err)
	}
	buf = bufio.NewWriter(fh)

	// Rotate our handles
	s.fh = fh
	s.buffered = buf
	s.offset = offset
	s.lastFlush = time.Now()
	return nil
}

// replay is used to seek to reset our internal state by replaying
// the snapshot file. It is used at initialization time to read old
// state
func (s *Snapshotter) replay() error {
	// Seek to the beginning
	if _, err := s.fh.Seek(0, os.SEEK_SET); err != nil {
		return err
	}

	// Read each line
	reader := bufio.NewReader(s.fh)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}

		// Skip the newline
		line = line[:len(line)-1]

		// Switch on the prefix
		if strings.HasPrefix(line, "alive: ") {
			info := strings.TrimPrefix(line, "alive: ")
			addrIdx := strings.LastIndex(info, " ")
			if addrIdx == -1 {
				s.logger.Printf("[WARN] serf: Failed to parse address: %v", line)
				continue
			}
			addr := info[addrIdx+1:]
			name := info[:addrIdx]
			s.aliveNodes[name] = addr

		} else if strings.HasPrefix(line, "not-alive: ") {
			name := strings.TrimPrefix(line, "not-alive: ")
			delete(s.aliveNodes, name)

		} else if strings.HasPrefix(line, "clock: ") {
			timeStr := strings.TrimPrefix(line, "clock: ")
			timeInt, err := strconv.ParseUint(timeStr, 10, 64)
			if err != nil {
				s.logger.Printf("[WARN] serf: Failed to convert clock time: %v", err)
				continue
			}
			s.lastClock = LamportTime(timeInt)

		} else if strings.HasPrefix(line, "event-clock: ") {
			timeStr := strings.TrimPrefix(line, "event-clock: ")
			timeInt, err := strconv.ParseUint(timeStr, 10, 64)
			if err != nil {
				s.logger.Printf("[WARN] serf: Failed to convert event clock time: %v", err)
				continue
			}
			s.lastEventClock = LamportTime(timeInt)

		} else if strings.HasPrefix(line, "query-clock: ") {
			timeStr := strings.TrimPrefix(line, "query-clock: ")
			timeInt, err := strconv.ParseUint(timeStr, 10, 64)
			if err != nil {
				s.logger.Printf("[WARN] serf: Failed to convert query clock time: %v", err)
				continue
			}
			s.lastQueryClock = LamportTime(timeInt)

		} else if strings.HasPrefix(line, "coordinate: ") {
			if s.coordClient == nil {
				s.logger.Printf("[WARN] serf: Ignoring snapshot coordinates since they are disabled")
				continue
			}

			coordStr := strings.TrimPrefix(line, "coordinate: ")
			var coord coordinate.Coordinate
			err := json.Unmarshal([]byte(coordStr), &coord)
			if err != nil {
				s.logger.Printf("[WARN] serf: Failed to decode coordinate: %v", err)
				continue
			}
			s.coordClient.SetCoordinate(&coord)
		} else if line == "leave" {
			// Ignore a leave if we plan on re-joining
			if s.rejoinAfterLeave {
				s.logger.Printf("[INFO] serf: Ignoring previous leave in snapshot")
				continue
			}
			s.aliveNodes = make(map[string]string)
			s.lastClock = 0
			s.lastEventClock = 0
			s.lastQueryClock = 0

		} else if strings.HasPrefix(line, "#") {
			// Skip comment lines

		} else {
			s.logger.Printf("[WARN] serf: Unrecognized snapshot line: %v", line)
		}
	}

	// Seek to the end
	if _, err := s.fh.Seek(0, os.SEEK_END); err != nil {
		return err
	}
	return nil
}
