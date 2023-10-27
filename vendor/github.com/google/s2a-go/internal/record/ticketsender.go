/*
 *
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package record

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/s2a-go/internal/handshaker/service"
	commonpb "github.com/google/s2a-go/internal/proto/common_go_proto"
	s2apb "github.com/google/s2a-go/internal/proto/s2a_go_proto"
	"github.com/google/s2a-go/internal/tokenmanager"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
)

// sessionTimeout is the timeout for creating a session with the S2A handshaker
// service.
const sessionTimeout = time.Second * 5

// s2aTicketSender sends session tickets to the S2A handshaker service.
type s2aTicketSender interface {
	// sendTicketsToS2A sends the given session tickets to the S2A handshaker
	// service.
	sendTicketsToS2A(sessionTickets [][]byte, callComplete chan bool)
}

// ticketStream is the stream used to send and receive session information.
type ticketStream interface {
	Send(*s2apb.SessionReq) error
	Recv() (*s2apb.SessionResp, error)
}

type ticketSender struct {
	// hsAddr stores the address of the S2A handshaker service.
	hsAddr string
	// connectionID is the connection identifier that was created and sent by
	// S2A at the end of a handshake.
	connectionID uint64
	// localIdentity is the local identity that was used by S2A during session
	// setup and included in the session result.
	localIdentity *commonpb.Identity
	// tokenManager manages access tokens for authenticating to S2A.
	tokenManager tokenmanager.AccessTokenManager
	// ensureProcessSessionTickets allows users to wait and ensure that all
	// available session tickets are sent to S2A before a process completes.
	ensureProcessSessionTickets *sync.WaitGroup
}

// sendTicketsToS2A sends the given sessionTickets to the S2A handshaker
// service. This is done asynchronously and writes to the error logs if an error
// occurs.
func (t *ticketSender) sendTicketsToS2A(sessionTickets [][]byte, callComplete chan bool) {
	// Note that the goroutine is in the function rather than at the caller
	// because the fake ticket sender used for testing must run synchronously
	// so that the session tickets can be accessed from it after the tests have
	// been run.
	if t.ensureProcessSessionTickets != nil {
		t.ensureProcessSessionTickets.Add(1)
	}
	go func() {
		if err := func() error {
			defer func() {
				if t.ensureProcessSessionTickets != nil {
					t.ensureProcessSessionTickets.Done()
				}
			}()
			hsConn, err := service.Dial(t.hsAddr)
			if err != nil {
				return err
			}
			client := s2apb.NewS2AServiceClient(hsConn)
			ctx, cancel := context.WithTimeout(context.Background(), sessionTimeout)
			defer cancel()
			session, err := client.SetUpSession(ctx)
			if err != nil {
				return err
			}
			defer func() {
				if err := session.CloseSend(); err != nil {
					grpclog.Error(err)
				}
			}()
			return t.writeTicketsToStream(session, sessionTickets)
		}(); err != nil {
			grpclog.Errorf("failed to send resumption tickets to S2A with identity: %v, %v",
				t.localIdentity, err)
		}
		callComplete <- true
		close(callComplete)
	}()
}

// writeTicketsToStream writes the given session tickets to the given stream.
func (t *ticketSender) writeTicketsToStream(stream ticketStream, sessionTickets [][]byte) error {
	if err := stream.Send(
		&s2apb.SessionReq{
			ReqOneof: &s2apb.SessionReq_ResumptionTicket{
				ResumptionTicket: &s2apb.ResumptionTicketReq{
					InBytes:       sessionTickets,
					ConnectionId:  t.connectionID,
					LocalIdentity: t.localIdentity,
				},
			},
			AuthMechanisms: t.getAuthMechanisms(),
		},
	); err != nil {
		return err
	}
	sessionResp, err := stream.Recv()
	if err != nil {
		return err
	}
	if sessionResp.GetStatus().GetCode() != uint32(codes.OK) {
		return fmt.Errorf("s2a session ticket response had error status: %v, %v",
			sessionResp.GetStatus().GetCode(), sessionResp.GetStatus().GetDetails())
	}
	return nil
}

func (t *ticketSender) getAuthMechanisms() []*s2apb.AuthenticationMechanism {
	if t.tokenManager == nil {
		return nil
	}
	// First handle the special case when no local identity has been provided
	// by the application. In this case, an AuthenticationMechanism with no local
	// identity will be sent.
	if t.localIdentity == nil {
		token, err := t.tokenManager.DefaultToken()
		if err != nil {
			grpclog.Infof("unable to get token for empty local identity: %v", err)
			return nil
		}
		return []*s2apb.AuthenticationMechanism{
			{
				MechanismOneof: &s2apb.AuthenticationMechanism_Token{
					Token: token,
				},
			},
		}
	}

	// Next, handle the case where the application (or the S2A) has specified
	// a local identity.
	token, err := t.tokenManager.Token(t.localIdentity)
	if err != nil {
		grpclog.Infof("unable to get token for local identity %v: %v", t.localIdentity, err)
		return nil
	}
	return []*s2apb.AuthenticationMechanism{
		{
			Identity: t.localIdentity,
			MechanismOneof: &s2apb.AuthenticationMechanism_Token{
				Token: token,
			},
		},
	}
}
