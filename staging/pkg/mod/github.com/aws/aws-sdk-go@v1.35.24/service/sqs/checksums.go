package sqs

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

var (
	errChecksumMissingBody = fmt.Errorf("cannot compute checksum. missing body")
	errChecksumMissingMD5  = fmt.Errorf("cannot verify checksum. missing response MD5")
)

func setupChecksumValidation(r *request.Request) {
	if aws.BoolValue(r.Config.DisableComputeChecksums) {
		return
	}

	switch r.Operation.Name {
	case opSendMessage:
		r.Handlers.Unmarshal.PushBack(verifySendMessage)
	case opSendMessageBatch:
		r.Handlers.Unmarshal.PushBack(verifySendMessageBatch)
	case opReceiveMessage:
		r.Handlers.Unmarshal.PushBack(verifyReceiveMessage)
	}
}

func verifySendMessage(r *request.Request) {
	if r.DataFilled() && r.ParamsFilled() {
		in := r.Params.(*SendMessageInput)
		out := r.Data.(*SendMessageOutput)
		err := checksumsMatch(in.MessageBody, out.MD5OfMessageBody)
		if err != nil {
			setChecksumError(r, err.Error())
		}
	}
}

func verifySendMessageBatch(r *request.Request) {
	if r.DataFilled() && r.ParamsFilled() {
		entries := map[string]*SendMessageBatchResultEntry{}
		ids := []string{}

		out := r.Data.(*SendMessageBatchOutput)
		for _, entry := range out.Successful {
			entries[*entry.Id] = entry
		}

		in := r.Params.(*SendMessageBatchInput)
		for _, entry := range in.Entries {
			if e, ok := entries[*entry.Id]; ok {
				if err := checksumsMatch(entry.MessageBody, e.MD5OfMessageBody); err != nil {
					ids = append(ids, *e.MessageId)
				}
			}
		}
		if len(ids) > 0 {
			setChecksumError(r, "invalid messages: %s", strings.Join(ids, ", "))
		}
	}
}

func verifyReceiveMessage(r *request.Request) {
	if r.DataFilled() && r.ParamsFilled() {
		ids := []string{}
		out := r.Data.(*ReceiveMessageOutput)
		for i, msg := range out.Messages {
			err := checksumsMatch(msg.Body, msg.MD5OfBody)
			if err != nil {
				if msg.MessageId == nil {
					if r.Config.Logger != nil {
						r.Config.Logger.Log(fmt.Sprintf(
							"WARN: SQS.ReceiveMessage failed checksum request id: %s, message %d has no message ID.",
							r.RequestID, i,
						))
					}
					continue
				}

				ids = append(ids, *msg.MessageId)
			}
		}
		if len(ids) > 0 {
			setChecksumError(r, "invalid messages: %s", strings.Join(ids, ", "))
		}
	}
}

func checksumsMatch(body, expectedMD5 *string) error {
	if body == nil {
		return errChecksumMissingBody
	} else if expectedMD5 == nil {
		return errChecksumMissingMD5
	}

	msum := md5.Sum([]byte(*body))
	sum := hex.EncodeToString(msum[:])
	if sum != *expectedMD5 {
		return fmt.Errorf("expected MD5 checksum '%s', got '%s'", *expectedMD5, sum)
	}

	return nil
}

func setChecksumError(r *request.Request, format string, args ...interface{}) {
	r.Retryable = aws.Bool(true)
	r.Error = awserr.New("InvalidChecksum", fmt.Sprintf(format, args...), nil)
}
