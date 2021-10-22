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

//go:generate ./regenerate.sh

// Package service provides an implementation for channelz service server.
package service

import (
	"context"
	"net"
	"time"

	"github.com/golang/protobuf/ptypes"
	durpb "github.com/golang/protobuf/ptypes/duration"
	wrpb "github.com/golang/protobuf/ptypes/wrappers"
	"google.golang.org/grpc"
	channelzgrpc "google.golang.org/grpc/channelz/grpc_channelz_v1"
	channelzpb "google.golang.org/grpc/channelz/grpc_channelz_v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/status"
)

func init() {
	channelz.TurnOn()
}

func convertToPtypesDuration(sec int64, usec int64) *durpb.Duration {
	return ptypes.DurationProto(time.Duration(sec*1e9 + usec*1e3))
}

// RegisterChannelzServiceToServer registers the channelz service to the given server.
func RegisterChannelzServiceToServer(s *grpc.Server) {
	channelzgrpc.RegisterChannelzServer(s, newCZServer())
}

func newCZServer() channelzgrpc.ChannelzServer {
	return &serverImpl{}
}

type serverImpl struct{}

func connectivityStateToProto(s connectivity.State) *channelzpb.ChannelConnectivityState {
	switch s {
	case connectivity.Idle:
		return &channelzpb.ChannelConnectivityState{State: channelzpb.ChannelConnectivityState_IDLE}
	case connectivity.Connecting:
		return &channelzpb.ChannelConnectivityState{State: channelzpb.ChannelConnectivityState_CONNECTING}
	case connectivity.Ready:
		return &channelzpb.ChannelConnectivityState{State: channelzpb.ChannelConnectivityState_READY}
	case connectivity.TransientFailure:
		return &channelzpb.ChannelConnectivityState{State: channelzpb.ChannelConnectivityState_TRANSIENT_FAILURE}
	case connectivity.Shutdown:
		return &channelzpb.ChannelConnectivityState{State: channelzpb.ChannelConnectivityState_SHUTDOWN}
	default:
		return &channelzpb.ChannelConnectivityState{State: channelzpb.ChannelConnectivityState_UNKNOWN}
	}
}

func channelTraceToProto(ct *channelz.ChannelTrace) *channelzpb.ChannelTrace {
	pbt := &channelzpb.ChannelTrace{}
	pbt.NumEventsLogged = ct.EventNum
	if ts, err := ptypes.TimestampProto(ct.CreationTime); err == nil {
		pbt.CreationTimestamp = ts
	}
	var events []*channelzpb.ChannelTraceEvent
	for _, e := range ct.Events {
		cte := &channelzpb.ChannelTraceEvent{
			Description: e.Desc,
			Severity:    channelzpb.ChannelTraceEvent_Severity(e.Severity),
		}
		if ts, err := ptypes.TimestampProto(e.Timestamp); err == nil {
			cte.Timestamp = ts
		}
		if e.RefID != 0 {
			switch e.RefType {
			case channelz.RefChannel:
				cte.ChildRef = &channelzpb.ChannelTraceEvent_ChannelRef{ChannelRef: &channelzpb.ChannelRef{ChannelId: e.RefID, Name: e.RefName}}
			case channelz.RefSubChannel:
				cte.ChildRef = &channelzpb.ChannelTraceEvent_SubchannelRef{SubchannelRef: &channelzpb.SubchannelRef{SubchannelId: e.RefID, Name: e.RefName}}
			}
		}
		events = append(events, cte)
	}
	pbt.Events = events
	return pbt
}

func channelMetricToProto(cm *channelz.ChannelMetric) *channelzpb.Channel {
	c := &channelzpb.Channel{}
	c.Ref = &channelzpb.ChannelRef{ChannelId: cm.ID, Name: cm.RefName}

	c.Data = &channelzpb.ChannelData{
		State:          connectivityStateToProto(cm.ChannelData.State),
		Target:         cm.ChannelData.Target,
		CallsStarted:   cm.ChannelData.CallsStarted,
		CallsSucceeded: cm.ChannelData.CallsSucceeded,
		CallsFailed:    cm.ChannelData.CallsFailed,
	}
	if ts, err := ptypes.TimestampProto(cm.ChannelData.LastCallStartedTimestamp); err == nil {
		c.Data.LastCallStartedTimestamp = ts
	}
	nestedChans := make([]*channelzpb.ChannelRef, 0, len(cm.NestedChans))
	for id, ref := range cm.NestedChans {
		nestedChans = append(nestedChans, &channelzpb.ChannelRef{ChannelId: id, Name: ref})
	}
	c.ChannelRef = nestedChans

	subChans := make([]*channelzpb.SubchannelRef, 0, len(cm.SubChans))
	for id, ref := range cm.SubChans {
		subChans = append(subChans, &channelzpb.SubchannelRef{SubchannelId: id, Name: ref})
	}
	c.SubchannelRef = subChans

	sockets := make([]*channelzpb.SocketRef, 0, len(cm.Sockets))
	for id, ref := range cm.Sockets {
		sockets = append(sockets, &channelzpb.SocketRef{SocketId: id, Name: ref})
	}
	c.SocketRef = sockets
	c.Data.Trace = channelTraceToProto(cm.Trace)
	return c
}

func subChannelMetricToProto(cm *channelz.SubChannelMetric) *channelzpb.Subchannel {
	sc := &channelzpb.Subchannel{}
	sc.Ref = &channelzpb.SubchannelRef{SubchannelId: cm.ID, Name: cm.RefName}

	sc.Data = &channelzpb.ChannelData{
		State:          connectivityStateToProto(cm.ChannelData.State),
		Target:         cm.ChannelData.Target,
		CallsStarted:   cm.ChannelData.CallsStarted,
		CallsSucceeded: cm.ChannelData.CallsSucceeded,
		CallsFailed:    cm.ChannelData.CallsFailed,
	}
	if ts, err := ptypes.TimestampProto(cm.ChannelData.LastCallStartedTimestamp); err == nil {
		sc.Data.LastCallStartedTimestamp = ts
	}
	nestedChans := make([]*channelzpb.ChannelRef, 0, len(cm.NestedChans))
	for id, ref := range cm.NestedChans {
		nestedChans = append(nestedChans, &channelzpb.ChannelRef{ChannelId: id, Name: ref})
	}
	sc.ChannelRef = nestedChans

	subChans := make([]*channelzpb.SubchannelRef, 0, len(cm.SubChans))
	for id, ref := range cm.SubChans {
		subChans = append(subChans, &channelzpb.SubchannelRef{SubchannelId: id, Name: ref})
	}
	sc.SubchannelRef = subChans

	sockets := make([]*channelzpb.SocketRef, 0, len(cm.Sockets))
	for id, ref := range cm.Sockets {
		sockets = append(sockets, &channelzpb.SocketRef{SocketId: id, Name: ref})
	}
	sc.SocketRef = sockets
	sc.Data.Trace = channelTraceToProto(cm.Trace)
	return sc
}

func securityToProto(se credentials.ChannelzSecurityValue) *channelzpb.Security {
	switch v := se.(type) {
	case *credentials.TLSChannelzSecurityValue:
		return &channelzpb.Security{Model: &channelzpb.Security_Tls_{Tls: &channelzpb.Security_Tls{
			CipherSuite:       &channelzpb.Security_Tls_StandardName{StandardName: v.StandardName},
			LocalCertificate:  v.LocalCertificate,
			RemoteCertificate: v.RemoteCertificate,
		}}}
	case *credentials.OtherChannelzSecurityValue:
		otherSecurity := &channelzpb.Security_OtherSecurity{
			Name: v.Name,
		}
		if anyval, err := ptypes.MarshalAny(v.Value); err == nil {
			otherSecurity.Value = anyval
		}
		return &channelzpb.Security{Model: &channelzpb.Security_Other{Other: otherSecurity}}
	}
	return nil
}

func addrToProto(a net.Addr) *channelzpb.Address {
	switch a.Network() {
	case "udp":
		// TODO: Address_OtherAddress{}. Need proto def for Value.
	case "ip":
		// Note zone info is discarded through the conversion.
		return &channelzpb.Address{Address: &channelzpb.Address_TcpipAddress{TcpipAddress: &channelzpb.Address_TcpIpAddress{IpAddress: a.(*net.IPAddr).IP}}}
	case "ip+net":
		// Note mask info is discarded through the conversion.
		return &channelzpb.Address{Address: &channelzpb.Address_TcpipAddress{TcpipAddress: &channelzpb.Address_TcpIpAddress{IpAddress: a.(*net.IPNet).IP}}}
	case "tcp":
		// Note zone info is discarded through the conversion.
		return &channelzpb.Address{Address: &channelzpb.Address_TcpipAddress{TcpipAddress: &channelzpb.Address_TcpIpAddress{IpAddress: a.(*net.TCPAddr).IP, Port: int32(a.(*net.TCPAddr).Port)}}}
	case "unix", "unixgram", "unixpacket":
		return &channelzpb.Address{Address: &channelzpb.Address_UdsAddress_{UdsAddress: &channelzpb.Address_UdsAddress{Filename: a.String()}}}
	default:
	}
	return &channelzpb.Address{}
}

func socketMetricToProto(sm *channelz.SocketMetric) *channelzpb.Socket {
	s := &channelzpb.Socket{}
	s.Ref = &channelzpb.SocketRef{SocketId: sm.ID, Name: sm.RefName}

	s.Data = &channelzpb.SocketData{
		StreamsStarted:   sm.SocketData.StreamsStarted,
		StreamsSucceeded: sm.SocketData.StreamsSucceeded,
		StreamsFailed:    sm.SocketData.StreamsFailed,
		MessagesSent:     sm.SocketData.MessagesSent,
		MessagesReceived: sm.SocketData.MessagesReceived,
		KeepAlivesSent:   sm.SocketData.KeepAlivesSent,
	}
	if ts, err := ptypes.TimestampProto(sm.SocketData.LastLocalStreamCreatedTimestamp); err == nil {
		s.Data.LastLocalStreamCreatedTimestamp = ts
	}
	if ts, err := ptypes.TimestampProto(sm.SocketData.LastRemoteStreamCreatedTimestamp); err == nil {
		s.Data.LastRemoteStreamCreatedTimestamp = ts
	}
	if ts, err := ptypes.TimestampProto(sm.SocketData.LastMessageSentTimestamp); err == nil {
		s.Data.LastMessageSentTimestamp = ts
	}
	if ts, err := ptypes.TimestampProto(sm.SocketData.LastMessageReceivedTimestamp); err == nil {
		s.Data.LastMessageReceivedTimestamp = ts
	}
	s.Data.LocalFlowControlWindow = &wrpb.Int64Value{Value: sm.SocketData.LocalFlowControlWindow}
	s.Data.RemoteFlowControlWindow = &wrpb.Int64Value{Value: sm.SocketData.RemoteFlowControlWindow}

	if sm.SocketData.SocketOptions != nil {
		s.Data.Option = sockoptToProto(sm.SocketData.SocketOptions)
	}
	if sm.SocketData.Security != nil {
		s.Security = securityToProto(sm.SocketData.Security)
	}

	if sm.SocketData.LocalAddr != nil {
		s.Local = addrToProto(sm.SocketData.LocalAddr)
	}
	if sm.SocketData.RemoteAddr != nil {
		s.Remote = addrToProto(sm.SocketData.RemoteAddr)
	}
	s.RemoteName = sm.SocketData.RemoteName
	return s
}

func (s *serverImpl) GetTopChannels(ctx context.Context, req *channelzpb.GetTopChannelsRequest) (*channelzpb.GetTopChannelsResponse, error) {
	metrics, end := channelz.GetTopChannels(req.GetStartChannelId(), req.GetMaxResults())
	resp := &channelzpb.GetTopChannelsResponse{}
	for _, m := range metrics {
		resp.Channel = append(resp.Channel, channelMetricToProto(m))
	}
	resp.End = end
	return resp, nil
}

func serverMetricToProto(sm *channelz.ServerMetric) *channelzpb.Server {
	s := &channelzpb.Server{}
	s.Ref = &channelzpb.ServerRef{ServerId: sm.ID, Name: sm.RefName}

	s.Data = &channelzpb.ServerData{
		CallsStarted:   sm.ServerData.CallsStarted,
		CallsSucceeded: sm.ServerData.CallsSucceeded,
		CallsFailed:    sm.ServerData.CallsFailed,
	}

	if ts, err := ptypes.TimestampProto(sm.ServerData.LastCallStartedTimestamp); err == nil {
		s.Data.LastCallStartedTimestamp = ts
	}
	sockets := make([]*channelzpb.SocketRef, 0, len(sm.ListenSockets))
	for id, ref := range sm.ListenSockets {
		sockets = append(sockets, &channelzpb.SocketRef{SocketId: id, Name: ref})
	}
	s.ListenSocket = sockets
	return s
}

func (s *serverImpl) GetServers(ctx context.Context, req *channelzpb.GetServersRequest) (*channelzpb.GetServersResponse, error) {
	metrics, end := channelz.GetServers(req.GetStartServerId(), req.GetMaxResults())
	resp := &channelzpb.GetServersResponse{}
	for _, m := range metrics {
		resp.Server = append(resp.Server, serverMetricToProto(m))
	}
	resp.End = end
	return resp, nil
}

func (s *serverImpl) GetServerSockets(ctx context.Context, req *channelzpb.GetServerSocketsRequest) (*channelzpb.GetServerSocketsResponse, error) {
	metrics, end := channelz.GetServerSockets(req.GetServerId(), req.GetStartSocketId(), req.GetMaxResults())
	resp := &channelzpb.GetServerSocketsResponse{}
	for _, m := range metrics {
		resp.SocketRef = append(resp.SocketRef, &channelzpb.SocketRef{SocketId: m.ID, Name: m.RefName})
	}
	resp.End = end
	return resp, nil
}

func (s *serverImpl) GetChannel(ctx context.Context, req *channelzpb.GetChannelRequest) (*channelzpb.GetChannelResponse, error) {
	var metric *channelz.ChannelMetric
	if metric = channelz.GetChannel(req.GetChannelId()); metric == nil {
		return nil, status.Errorf(codes.NotFound, "requested channel %d not found", req.GetChannelId())
	}
	resp := &channelzpb.GetChannelResponse{Channel: channelMetricToProto(metric)}
	return resp, nil
}

func (s *serverImpl) GetSubchannel(ctx context.Context, req *channelzpb.GetSubchannelRequest) (*channelzpb.GetSubchannelResponse, error) {
	var metric *channelz.SubChannelMetric
	if metric = channelz.GetSubChannel(req.GetSubchannelId()); metric == nil {
		return nil, status.Errorf(codes.NotFound, "requested sub channel %d not found", req.GetSubchannelId())
	}
	resp := &channelzpb.GetSubchannelResponse{Subchannel: subChannelMetricToProto(metric)}
	return resp, nil
}

func (s *serverImpl) GetSocket(ctx context.Context, req *channelzpb.GetSocketRequest) (*channelzpb.GetSocketResponse, error) {
	var metric *channelz.SocketMetric
	if metric = channelz.GetSocket(req.GetSocketId()); metric == nil {
		return nil, status.Errorf(codes.NotFound, "requested socket %d not found", req.GetSocketId())
	}
	resp := &channelzpb.GetSocketResponse{Socket: socketMetricToProto(metric)}
	return resp, nil
}

func (s *serverImpl) GetServer(ctx context.Context, req *channelzpb.GetServerRequest) (*channelzpb.GetServerResponse, error) {
	var metric *channelz.ServerMetric
	if metric = channelz.GetServer(req.GetServerId()); metric == nil {
		return nil, status.Errorf(codes.NotFound, "requested server %d not found", req.GetServerId())
	}
	resp := &channelzpb.GetServerResponse{Server: serverMetricToProto(metric)}
	return resp, nil
}
