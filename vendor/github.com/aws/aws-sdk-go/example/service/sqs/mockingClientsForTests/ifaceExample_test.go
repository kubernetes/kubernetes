// +build example

package main

import (
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/sqs"
	"github.com/aws/aws-sdk-go/service/sqs/sqsiface"
)

type mockedReceiveMsgs struct {
	sqsiface.SQSAPI
	Resp sqs.ReceiveMessageOutput
}

func (m mockedReceiveMsgs) ReceiveMessage(in *sqs.ReceiveMessageInput) (*sqs.ReceiveMessageOutput, error) {
	// Only need to return mocked response output
	return &m.Resp, nil
}

func TestQueueGetMessage(t *testing.T) {
	cases := []struct {
		Resp     sqs.ReceiveMessageOutput
		Expected []Message
	}{
		{ // Case 1, expect parsed responses
			Resp: sqs.ReceiveMessageOutput{
				Messages: []*sqs.Message{
					{Body: aws.String(`{"from":"user_1","to":"room_1","msg":"Hello!"}`)},
					{Body: aws.String(`{"from":"user_2","to":"room_1","msg":"Hi user_1 :)"}`)},
				},
			},
			Expected: []Message{
				{From: "user_1", To: "room_1", Msg: "Hello!"},
				{From: "user_2", To: "room_1", Msg: "Hi user_1 :)"},
			},
		},
		{ // Case 2, not messages returned
			Resp:     sqs.ReceiveMessageOutput{},
			Expected: []Message{},
		},
	}

	for i, c := range cases {
		q := Queue{
			Client: mockedReceiveMsgs{Resp: c.Resp},
			URL:    fmt.Sprintf("mockURL_%d", i),
		}
		msgs, err := q.GetMessages(20)
		if err != nil {
			t.Fatalf("%d, unexpected error, %v", i, err)
		}
		if a, e := len(msgs), len(c.Expected); a != e {
			t.Fatalf("%d, expected %d messages, got %d", i, e, a)
		}
		for j, msg := range msgs {
			if a, e := msg, c.Expected[j]; a != e {
				t.Errorf("%d, expected %v message, got %v", i, e, a)
			}
		}
	}
}
