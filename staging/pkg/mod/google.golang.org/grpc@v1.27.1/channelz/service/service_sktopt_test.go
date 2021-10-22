// +build linux,!appengine
// +build 386 amd64

/*
 *
 * Copyright 2018 gRPC authors.
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

// SocketOptions is only supported on linux system. The functions defined in
// this file are to parse the socket option field and the test is specifically
// to verify the behavior of socket option parsing.

package service

import (
	"context"
	"reflect"
	"strconv"
	"testing"

	"github.com/golang/protobuf/ptypes"
	durpb "github.com/golang/protobuf/ptypes/duration"
	"golang.org/x/sys/unix"
	channelzpb "google.golang.org/grpc/channelz/grpc_channelz_v1"
	"google.golang.org/grpc/internal/channelz"
)

func init() {
	// Assign protoToSocketOption to protoToSocketOpt in order to enable socket option
	// data conversion from proto message to channelz defined struct.
	protoToSocketOpt = protoToSocketOption
}

func convertToDuration(d *durpb.Duration) (sec int64, usec int64) {
	if d != nil {
		if dur, err := ptypes.Duration(d); err == nil {
			sec = int64(int64(dur) / 1e9)
			usec = (int64(dur) - sec*1e9) / 1e3
		}
	}
	return
}

func protoToLinger(protoLinger *channelzpb.SocketOptionLinger) *unix.Linger {
	linger := &unix.Linger{}
	if protoLinger.GetActive() {
		linger.Onoff = 1
	}
	lv, _ := convertToDuration(protoLinger.GetDuration())
	linger.Linger = int32(lv)
	return linger
}

func protoToSocketOption(skopts []*channelzpb.SocketOption) *channelz.SocketOptionData {
	skdata := &channelz.SocketOptionData{}
	for _, opt := range skopts {
		switch opt.GetName() {
		case "SO_LINGER":
			protoLinger := &channelzpb.SocketOptionLinger{}
			err := ptypes.UnmarshalAny(opt.GetAdditional(), protoLinger)
			if err == nil {
				skdata.Linger = protoToLinger(protoLinger)
			}
		case "SO_RCVTIMEO":
			protoTimeout := &channelzpb.SocketOptionTimeout{}
			err := ptypes.UnmarshalAny(opt.GetAdditional(), protoTimeout)
			if err == nil {
				skdata.RecvTimeout = protoToTime(protoTimeout)
			}
		case "SO_SNDTIMEO":
			protoTimeout := &channelzpb.SocketOptionTimeout{}
			err := ptypes.UnmarshalAny(opt.GetAdditional(), protoTimeout)
			if err == nil {
				skdata.SendTimeout = protoToTime(protoTimeout)
			}
		case "TCP_INFO":
			tcpi := &channelzpb.SocketOptionTcpInfo{}
			err := ptypes.UnmarshalAny(opt.GetAdditional(), tcpi)
			if err == nil {
				skdata.TCPInfo = &unix.TCPInfo{
					State:          uint8(tcpi.TcpiState),
					Ca_state:       uint8(tcpi.TcpiCaState),
					Retransmits:    uint8(tcpi.TcpiRetransmits),
					Probes:         uint8(tcpi.TcpiProbes),
					Backoff:        uint8(tcpi.TcpiBackoff),
					Options:        uint8(tcpi.TcpiOptions),
					Rto:            tcpi.TcpiRto,
					Ato:            tcpi.TcpiAto,
					Snd_mss:        tcpi.TcpiSndMss,
					Rcv_mss:        tcpi.TcpiRcvMss,
					Unacked:        tcpi.TcpiUnacked,
					Sacked:         tcpi.TcpiSacked,
					Lost:           tcpi.TcpiLost,
					Retrans:        tcpi.TcpiRetrans,
					Fackets:        tcpi.TcpiFackets,
					Last_data_sent: tcpi.TcpiLastDataSent,
					Last_ack_sent:  tcpi.TcpiLastAckSent,
					Last_data_recv: tcpi.TcpiLastDataRecv,
					Last_ack_recv:  tcpi.TcpiLastAckRecv,
					Pmtu:           tcpi.TcpiPmtu,
					Rcv_ssthresh:   tcpi.TcpiRcvSsthresh,
					Rtt:            tcpi.TcpiRtt,
					Rttvar:         tcpi.TcpiRttvar,
					Snd_ssthresh:   tcpi.TcpiSndSsthresh,
					Snd_cwnd:       tcpi.TcpiSndCwnd,
					Advmss:         tcpi.TcpiAdvmss,
					Reordering:     tcpi.TcpiReordering}
			}
		}
	}
	return skdata
}

func TestGetSocketOptions(t *testing.T) {
	czCleanup := channelz.NewChannelzStorage()
	defer cleanupWrapper(czCleanup, t)
	ss := []*dummySocket{
		{
			socketOptions: &channelz.SocketOptionData{
				Linger:      &unix.Linger{Onoff: 1, Linger: 2},
				RecvTimeout: &unix.Timeval{Sec: 10, Usec: 1},
				SendTimeout: &unix.Timeval{},
				TCPInfo:     &unix.TCPInfo{State: 1},
			},
		},
	}
	svr := newCZServer()
	ids := make([]int64, len(ss))
	svrID := channelz.RegisterServer(&dummyServer{}, "")
	defer channelz.RemoveEntry(svrID)
	for i, s := range ss {
		ids[i] = channelz.RegisterNormalSocket(s, svrID, strconv.Itoa(i))
		defer channelz.RemoveEntry(ids[i])
	}
	for i, s := range ss {
		resp, _ := svr.GetSocket(context.Background(), &channelzpb.GetSocketRequest{SocketId: ids[i]})
		metrics := resp.GetSocket()
		if !reflect.DeepEqual(metrics.GetRef(), &channelzpb.SocketRef{SocketId: ids[i], Name: strconv.Itoa(i)}) || !reflect.DeepEqual(socketProtoToStruct(metrics), s) {
			t.Fatalf("resp.GetSocket() want: metrics.GetRef() = %#v and %#v, got: metrics.GetRef() = %#v and %#v", &channelzpb.SocketRef{SocketId: ids[i], Name: strconv.Itoa(i)}, s, metrics.GetRef(), socketProtoToStruct(metrics))
		}
	}
}
