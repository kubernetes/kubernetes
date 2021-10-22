/*
 *
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"google.golang.org/grpc/grpclog"
	ppb "google.golang.org/grpc/profiling/proto"
)

type jsonNode struct {
	Name      string  `json:"name"`
	Cat       string  `json:"cat"`
	ID        string  `json:"id"`
	Cname     string  `json:"cname"`
	Phase     string  `json:"ph"`
	Timestamp float64 `json:"ts"`
	PID       string  `json:"pid"`
	TID       string  `json:"tid"`
}

// Catapult does not allow specifying colours manually; a 20-odd predefined
// labels are used (that don't make much sense outside the context of
// Chromium). See this for more details:
//
// https://github.com/catapult-project/catapult/blob/bef344f7017fc9e04f7049d0f58af6d9ce9f4ab6/tracing/tracing/base/color_scheme.html#L29
func hashCname(tag string) string {
	if strings.Contains(tag, "encoding") {
		return "rail_response"
	}

	if strings.Contains(tag, "compression") {
		return "cq_build_passed"
	}

	if strings.Contains(tag, "transport") {
		if strings.Contains(tag, "blocking") {
			return "rail_animation"
		}
		return "good"
	}

	if strings.Contains(tag, "header") {
		return "cq_build_attempt_failed"
	}

	if tag == "/" {
		return "heap_dump_stack_frame"
	}

	if strings.Contains(tag, "flow") || strings.Contains(tag, "tmp") {
		return "heap_dump_stack_frame"
	}

	return ""
}

// filterCounter identifies the counter-th instance of a timer of the type
// `filter` within a Stat. This, in conjunction with the counter data structure
// defined below, is used to draw flows between linked loopy writer/reader
// events with application goroutine events in trace-viewer. This is possible
// because enqueues and dequeues are ordered -- that is, the first dequeue must
// be dequeueing the first enqueue operation.
func filterCounter(stat *ppb.Stat, filter string, counter int) int {
	localCounter := 0
	for i := 0; i < len(stat.Timers); i++ {
		if stat.Timers[i].Tags == filter {
			if localCounter == counter {
				return i
			}
			localCounter++
		}
	}

	return -1
}

// counter is state object used to store and retrieve the number of timers of a
// particular type that have been seen.
type counter struct {
	c map[string]int
}

func newCounter() *counter {
	return &counter{c: make(map[string]int)}
}

func (c *counter) GetAndInc(s string) int {
	ret := c.c[s]
	c.c[s]++
	return ret
}

func catapultNs(sec int64, nsec int32) float64 {
	return float64((sec * 1000000000) + int64(nsec))
}

// streamStatsCatapultJSONSingle processes a single proto Stat object to return
// an array of jsonNodes in trace-viewer's format.
func streamStatsCatapultJSONSingle(stat *ppb.Stat, baseSec int64, baseNsec int32) []jsonNode {
	if len(stat.Timers) == 0 {
		return nil
	}

	connectionCounter := binary.BigEndian.Uint64(stat.Metadata[0:8])
	streamID := binary.BigEndian.Uint32(stat.Metadata[8:12])
	opid := fmt.Sprintf("/%s/%d/%d", stat.Tags, connectionCounter, streamID)

	var loopyReaderGoID, loopyWriterGoID int64
	for i := 0; i < len(stat.Timers) && (loopyReaderGoID == 0 || loopyWriterGoID == 0); i++ {
		if strings.Contains(stat.Timers[i].Tags, "/loopyReader") {
			loopyReaderGoID = stat.Timers[i].GoId
		} else if strings.Contains(stat.Timers[i].Tags, "/loopyWriter") {
			loopyWriterGoID = stat.Timers[i].GoId
		}
	}

	lrc, lwc := newCounter(), newCounter()

	var result []jsonNode
	result = append(result,
		jsonNode{
			Name:      "loopyReaderTmp",
			ID:        opid,
			Cname:     hashCname("tmp"),
			Phase:     "i",
			Timestamp: 0,
			PID:       fmt.Sprintf("/%s/%d/loopyReader", stat.Tags, connectionCounter),
			TID:       fmt.Sprintf("%d", loopyReaderGoID),
		},
		jsonNode{
			Name:      "loopyWriterTmp",
			ID:        opid,
			Cname:     hashCname("tmp"),
			Phase:     "i",
			Timestamp: 0,
			PID:       fmt.Sprintf("/%s/%d/loopyWriter", stat.Tags, connectionCounter),
			TID:       fmt.Sprintf("%d", loopyWriterGoID),
		},
	)

	for i := 0; i < len(stat.Timers); i++ {
		categories := stat.Tags
		pid, tid := opid, fmt.Sprintf("%d", stat.Timers[i].GoId)

		if stat.Timers[i].GoId == loopyReaderGoID {
			pid, tid = fmt.Sprintf("/%s/%d/loopyReader", stat.Tags, connectionCounter), fmt.Sprintf("%d", stat.Timers[i].GoId)

			var flowEndID int
			var flowEndPID, flowEndTID string
			switch stat.Timers[i].Tags {
			case "/http2/recv/header":
				flowEndID = filterCounter(stat, "/grpc/stream/recv/header", lrc.GetAndInc("/http2/recv/header"))
				if flowEndID != -1 {
					flowEndPID = opid
					flowEndTID = fmt.Sprintf("%d", stat.Timers[flowEndID].GoId)
				} else {
					grpclog.Infof("cannot find %s/grpc/stream/recv/header for %s/http2/recv/header", opid, opid)
				}
			case "/http2/recv/dataFrame/loopyReader":
				flowEndID = filterCounter(stat, "/recvAndDecompress", lrc.GetAndInc("/http2/recv/dataFrame/loopyReader"))
				if flowEndID != -1 {
					flowEndPID = opid
					flowEndTID = fmt.Sprintf("%d", stat.Timers[flowEndID].GoId)
				} else {
					grpclog.Infof("cannot find %s/recvAndDecompress for %s/http2/recv/dataFrame/loopyReader", opid, opid)
				}
			default:
				flowEndID = -1
			}

			if flowEndID != -1 {
				flowID := fmt.Sprintf("lrc begin:/%d%s end:/%d%s begin:(%d, %s, %s) end:(%d, %s, %s)", connectionCounter, stat.Timers[i].Tags, connectionCounter, stat.Timers[flowEndID].Tags, i, pid, tid, flowEndID, flowEndPID, flowEndTID)
				result = append(result,
					jsonNode{
						Name:      fmt.Sprintf("%s/flow", opid),
						Cat:       categories + ",flow",
						ID:        flowID,
						Cname:     hashCname("flow"),
						Phase:     "s",
						Timestamp: catapultNs(stat.Timers[i].EndSec-baseSec, stat.Timers[i].EndNsec-baseNsec),
						PID:       pid,
						TID:       tid,
					},
					jsonNode{
						Name:      fmt.Sprintf("%s/flow", opid),
						Cat:       categories + ",flow",
						ID:        flowID,
						Cname:     hashCname("flow"),
						Phase:     "f",
						Timestamp: catapultNs(stat.Timers[flowEndID].BeginSec-baseSec, stat.Timers[flowEndID].BeginNsec-baseNsec),
						PID:       flowEndPID,
						TID:       flowEndTID,
					},
				)
			}
		} else if stat.Timers[i].GoId == loopyWriterGoID {
			pid, tid = fmt.Sprintf("/%s/%d/loopyWriter", stat.Tags, connectionCounter), fmt.Sprintf("%d", stat.Timers[i].GoId)

			var flowBeginID int
			var flowBeginPID, flowBeginTID string
			switch stat.Timers[i].Tags {
			case "/http2/recv/header/loopyWriter/registerOutStream":
				flowBeginID = filterCounter(stat, "/http2/recv/header", lwc.GetAndInc("/http2/recv/header/loopyWriter/registerOutStream"))
				flowBeginPID = fmt.Sprintf("/%s/%d/loopyReader", stat.Tags, connectionCounter)
				flowBeginTID = fmt.Sprintf("%d", loopyReaderGoID)
			case "/http2/send/dataFrame/loopyWriter/preprocess":
				flowBeginID = filterCounter(stat, "/transport/enqueue", lwc.GetAndInc("/http2/send/dataFrame/loopyWriter/preprocess"))
				if flowBeginID != -1 {
					flowBeginPID = opid
					flowBeginTID = fmt.Sprintf("%d", stat.Timers[flowBeginID].GoId)
				} else {
					grpclog.Infof("cannot find /%d/transport/enqueue for /%d/http2/send/dataFrame/loopyWriter/preprocess", connectionCounter, connectionCounter)
				}
			default:
				flowBeginID = -1
			}

			if flowBeginID != -1 {
				flowID := fmt.Sprintf("lwc begin:/%d%s end:/%d%s begin:(%d, %s, %s) end:(%d, %s, %s)", connectionCounter, stat.Timers[flowBeginID].Tags, connectionCounter, stat.Timers[i].Tags, flowBeginID, flowBeginPID, flowBeginTID, i, pid, tid)
				result = append(result,
					jsonNode{
						Name:      fmt.Sprintf("/%s/%d/%d/flow", stat.Tags, connectionCounter, streamID),
						Cat:       categories + ",flow",
						ID:        flowID,
						Cname:     hashCname("flow"),
						Phase:     "s",
						Timestamp: catapultNs(stat.Timers[flowBeginID].EndSec-baseSec, stat.Timers[flowBeginID].EndNsec-baseNsec),
						PID:       flowBeginPID,
						TID:       flowBeginTID,
					},
					jsonNode{
						Name:      fmt.Sprintf("/%s/%d/%d/flow", stat.Tags, connectionCounter, streamID),
						Cat:       categories + ",flow",
						ID:        flowID,
						Cname:     hashCname("flow"),
						Phase:     "f",
						Timestamp: catapultNs(stat.Timers[i].BeginSec-baseSec, stat.Timers[i].BeginNsec-baseNsec),
						PID:       pid,
						TID:       tid,
					},
				)
			}
		}

		result = append(result,
			jsonNode{
				Name:      fmt.Sprintf("%s%s", opid, stat.Timers[i].Tags),
				Cat:       categories,
				ID:        opid,
				Cname:     hashCname(stat.Timers[i].Tags),
				Phase:     "B",
				Timestamp: catapultNs(stat.Timers[i].BeginSec-baseSec, stat.Timers[i].BeginNsec-baseNsec),
				PID:       pid,
				TID:       tid,
			},
			jsonNode{
				Name:      fmt.Sprintf("%s%s", opid, stat.Timers[i].Tags),
				Cat:       categories,
				ID:        opid,
				Cname:     hashCname(stat.Timers[i].Tags),
				Phase:     "E",
				Timestamp: catapultNs(stat.Timers[i].EndSec-baseSec, stat.Timers[i].EndNsec-baseNsec),
				PID:       pid,
				TID:       tid,
			},
		)
	}

	return result
}

// timerBeginIsBefore compares two proto Timer objects to determine if the
// first comes before the second chronologically.
func timerBeginIsBefore(ti *ppb.Timer, tj *ppb.Timer) bool {
	if ti.BeginSec == tj.BeginSec {
		return ti.BeginNsec < tj.BeginNsec
	}
	return ti.BeginSec < tj.BeginSec
}

// streamStatsCatapulJSON receives a *snapshot and the name of a JSON file to
// write to. The grpc-go profiling snapshot is processed and converted to a
// JSON format that can be understood by trace-viewer.
func streamStatsCatapultJSON(s *snapshot, streamStatsCatapultJSONFileName string) (err error) {
	grpclog.Infof("calculating stream stats filters")
	filterArray := strings.Split(*flagStreamStatsFilter, ",")
	filter := make(map[string]bool)
	for _, f := range filterArray {
		filter[f] = true
	}

	grpclog.Infof("filter stream stats for %s", *flagStreamStatsFilter)
	var streamStats []*ppb.Stat
	for _, stat := range s.StreamStats {
		if _, ok := filter[stat.Tags]; ok {
			streamStats = append(streamStats, stat)
		}
	}

	grpclog.Infof("sorting timers within all stats")
	for id := range streamStats {
		sort.Slice(streamStats[id].Timers, func(i, j int) bool {
			return timerBeginIsBefore(streamStats[id].Timers[i], streamStats[id].Timers[j])
		})
	}

	grpclog.Infof("sorting stream stats")
	sort.Slice(streamStats, func(i, j int) bool {
		if len(streamStats[j].Timers) == 0 {
			return true
		} else if len(streamStats[i].Timers) == 0 {
			return false
		}
		pi := binary.BigEndian.Uint64(streamStats[i].Metadata[0:8])
		pj := binary.BigEndian.Uint64(streamStats[j].Metadata[0:8])
		if pi == pj {
			return timerBeginIsBefore(streamStats[i].Timers[0], streamStats[j].Timers[0])
		}

		return pi < pj
	})

	// Clip the last stat as it's from the /Profiling/GetStreamStats call that we
	// made to retrieve the stats themselves. This likely happened millions of
	// nanoseconds after the last stream we want to profile, so it'd just make
	// the catapult graph less readable.
	if len(streamStats) > 0 {
		streamStats = streamStats[:len(streamStats)-1]
	}

	// All timestamps use the earliest timestamp available as the reference.
	grpclog.Infof("calculating the earliest timestamp across all timers")
	var base *ppb.Timer
	for _, stat := range streamStats {
		for _, timer := range stat.Timers {
			if base == nil || timerBeginIsBefore(base, timer) {
				base = timer
			}
		}
	}

	grpclog.Infof("converting %d stats to catapult JSON format", len(streamStats))
	var jsonNodes []jsonNode
	for _, stat := range streamStats {
		jsonNodes = append(jsonNodes, streamStatsCatapultJSONSingle(stat, base.BeginSec, base.BeginNsec)...)
	}

	grpclog.Infof("marshalling catapult JSON")
	b, err := json.Marshal(jsonNodes)
	if err != nil {
		grpclog.Errorf("cannot marshal JSON: %v", err)
		return err
	}

	grpclog.Infof("creating catapult JSON file")
	streamStatsCatapultJSONFile, err := os.Create(streamStatsCatapultJSONFileName)
	if err != nil {
		grpclog.Errorf("cannot create file %s: %v", streamStatsCatapultJSONFileName, err)
		return err
	}
	defer streamStatsCatapultJSONFile.Close()

	grpclog.Infof("writing catapult JSON to disk")
	_, err = streamStatsCatapultJSONFile.Write(b)
	if err != nil {
		grpclog.Errorf("cannot write marshalled JSON: %v", err)
		return err
	}

	grpclog.Infof("successfully wrote catapult JSON file %s", streamStatsCatapultJSONFileName)
	return nil
}
