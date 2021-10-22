// Copyright 2020 The Prometheus Authors
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

// +build !windows

package sysfs

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/prometheus/procfs/internal/util"
)

const fibrechannelClassPath = "class/fc_host"

type FibreChannelCounters struct {
	DumpedFrames          uint64 // /sys/class/fc_host/<Name>/statistics/dumped_frames
	ErrorFrames           uint64 // /sys/class/fc_host/<Name>/statistics/error_frames
	InvalidCRCCount       uint64 // /sys/class/fc_host/<Name>/statistics/invalid_crc_count
	RXFrames              uint64 // /sys/class/fc_host/<Name>/statistics/rx_frames
	RXWords               uint64 // /sys/class/fc_host/<Name>/statistics/rx_words
	TXFrames              uint64 // /sys/class/fc_host/<Name>/statistics/tx_frames
	TXWords               uint64 // /sys/class/fc_host/<Name>/statistics/tx_words
	SecondsSinceLastReset uint64 // /sys/class/fc_host/<Name>/statistics/seconds_since_last_reset
	InvalidTXWordCount    uint64 // /sys/class/fc_host/<Name>/statistics/invalid_tx_word_count
	LinkFailureCount      uint64 // /sys/class/fc_host/<Name>/statistics/link_failure_count
	LossOfSyncCount       uint64 // /sys/class/fc_host/<Name>/statistics/loss_of_sync_count
	LossOfSignalCount     uint64 // /sys/class/fc_host/<Name>/statistics/loss_of_signal_count
	NosCount              uint64 // /sys/class/fc_host/<Name>/statistics/nos_count
	FCPPacketAborts       uint64 // / sys/class/fc_host/<Name>/statistics/fcp_packet_aborts
}

type FibreChannelHost struct {
	Name             string               // /sys/class/fc_host/<Name>
	Speed            string               // /sys/class/fc_host/<Name>/speed
	PortState        string               // /sys/class/fc_host/<Name>/port_state
	PortType         string               // /sys/class/fc_host/<Name>/port_type
	SymbolicName     string               // /sys/class/fc_host/<Name>/symbolic_name
	NodeName         string               // /sys/class/fc_host/<Name>/node_name
	PortID           string               // /sys/class/fc_host/<Name>/port_id
	PortName         string               // /sys/class/fc_host/<Name>/port_name
	FabricName       string               // /sys/class/fc_host/<Name>/fabric_name
	DevLossTMO       string               // /sys/class/fc_host/<Name>/dev_loss_tmo
	SupportedClasses string               // /sys/class/fc_host/<Name>/supported_classes
	SupportedSpeeds  string               // /sys/class/fc_host/<Name>/supported_speeds
	Counters         FibreChannelCounters // /sys/class/fc_host/<Name>/statistics/*
}

type FibreChannelClass map[string]FibreChannelHost

// FibreChannelClass parses everything in /sys/class/fc_host.
func (fs FS) FibreChannelClass() (FibreChannelClass, error) {
	path := fs.sys.Path(fibrechannelClassPath)

	dirs, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}

	fcc := make(FibreChannelClass, len(dirs))
	for _, d := range dirs {
		host, err := fs.parseFibreChannelHost(d.Name())
		if err != nil {
			return nil, err
		}

		fcc[host.Name] = *host
	}

	return fcc, nil
}

// Parse a single FC host
func (fs FS) parseFibreChannelHost(name string) (*FibreChannelHost, error) {
	path := fs.sys.Path(fibrechannelClassPath, name)
	host := FibreChannelHost{Name: name}

	for _, f := range [...]string{"speed", "port_state", "port_type", "node_name", "port_id", "port_name", "fabric_name", "dev_loss_tmo", "symbolic_name", "supported_classes", "supported_speeds"} {
		name := filepath.Join(path, f)
		value, err := util.SysReadFile(name)
		if err != nil {
			return nil, fmt.Errorf("failed to read file %q: %v", name, err)
		}

		switch f {
		case "speed":
			host.Speed = value
		case "port_state":
			host.PortState = value
		case "port_type":
			host.PortType = value
		case "node_name":
			if len(value) > 2 {
				value = value[2:]
			}
			host.NodeName = value
		case "port_id":
			if len(value) > 2 {
				value = value[2:]
			}
			host.PortID = value
		case "port_name":
			if len(value) > 2 {
				value = value[2:]
			}
			host.PortName = value
		case "fabric_name":
			if len(value) > 2 {
				value = value[2:]
			}
			host.FabricName = value
		case "dev_loss_tmo":
			host.DevLossTMO = value
		case "supported_classes":
			host.SupportedClasses = value
		case "supported_speeds":
			host.SupportedSpeeds = value
		case "symbolic_name":
			host.SymbolicName = value
		}
	}

	counters, err := parseFibreChannelStatistics(path)
	if err != nil {
		return nil, err
	}
	host.Counters = *counters

	return &host, nil
}

// parseFibreChannelStatistics parses metrics from a single FC host.
func parseFibreChannelStatistics(hostPath string) (*FibreChannelCounters, error) {
	var counters FibreChannelCounters

	path := filepath.Join(hostPath, "statistics")
	files, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}

	for _, f := range files {
		if !f.Mode().IsRegular() || f.Name() == "reset_statistics" {
			continue
		}

		name := filepath.Join(path, f.Name())
		value, err := util.SysReadFile(name)
		if err != nil {
			// there are some write-only files in this directory; we can safely skip over them
			if os.IsNotExist(err) || err.Error() == "operation not supported" || err.Error() == "invalid argument" {
				continue
			}
			return nil, fmt.Errorf("failed to read file %q: %v", name, err)
		}

		vp := util.NewValueParser(value)

		// Below switch was automatically generated. Don't need everything in there yet, so the unwanted bits are commented out.
		switch f.Name() {
		case "dumped_frames":
			counters.DumpedFrames = *vp.PUInt64()
		case "error_frames":
			counters.ErrorFrames = *vp.PUInt64()
		/*
			case "fc_no_free_exch":
				counters.FcNoFreeExch = *vp.PUInt64()
			case "fc_no_free_exch_xid":
				counters.FcNoFreeExchXid = *vp.PUInt64()
			case "fc_non_bls_resp":
				counters.FcNonBlsResp = *vp.PUInt64()
			case "fc_seq_not_found":
				counters.FcSeqNotFound = *vp.PUInt64()
			case "fc_xid_busy":
				counters.FcXidBusy = *vp.PUInt64()
			case "fc_xid_not_found":
				counters.FcXidNotFound = *vp.PUInt64()
			case "fcp_control_requests":
				counters.FcpControlRequests = *vp.PUInt64()
			case "fcp_frame_alloc_failures":
				counters.FcpFrameAllocFailures = *vp.PUInt64()
			case "fcp_input_megabytes":
				counters.FcpInputMegabytes = *vp.PUInt64()
			case "fcp_input_requests":
				counters.FcpInputRequests = *vp.PUInt64()
			case "fcp_output_megabytes":
				counters.FcpOutputMegabytes = *vp.PUInt64()
			case "fcp_output_requests":
				counters.FcpOutputRequests = *vp.PUInt64()
		*/
		case "fcp_packet_aborts":
			counters.FCPPacketAborts = *vp.PUInt64()
			/*
				case "fcp_packet_alloc_failures":
					counters.FcpPacketAllocFailures = *vp.PUInt64()
			*/
		case "invalid_tx_word_count":
			counters.InvalidTXWordCount = *vp.PUInt64()
		case "invalid_crc_count":
			counters.InvalidCRCCount = *vp.PUInt64()
		case "link_failure_count":
			counters.LinkFailureCount = *vp.PUInt64()
		/*
			case "lip_count":
					counters.LipCount = *vp.PUInt64()
		*/
		case "loss_of_signal_count":
			counters.LossOfSignalCount = *vp.PUInt64()
		case "loss_of_sync_count":
			counters.LossOfSyncCount = *vp.PUInt64()
		case "nos_count":
			counters.NosCount = *vp.PUInt64()
		/*
			case "prim_seq_protocol_err_count":
				counters.PrimSeqProtocolErrCount = *vp.PUInt64()
		*/
		case "rx_frames":
			counters.RXFrames = *vp.PUInt64()
		case "rx_words":
			counters.RXWords = *vp.PUInt64()
		case "seconds_since_last_reset":
			counters.SecondsSinceLastReset = *vp.PUInt64()
		case "tx_frames":
			counters.TXFrames = *vp.PUInt64()
		case "tx_words":
			counters.TXWords = *vp.PUInt64()
		}

		if err := vp.Err(); err != nil {
			return nil, err
		}

	}

	return &counters, nil
}
