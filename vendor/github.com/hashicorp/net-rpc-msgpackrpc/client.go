package msgpackrpc

import (
	"errors"
	"net/rpc"
	"sync/atomic"

	"github.com/hashicorp/go-multierror"
)

var (
	// nextCallSeq is used to assign a unique sequence number
	// to each call made with CallWithCodec
	nextCallSeq uint64
)

// CallWithCodec is used to perform the same actions as rpc.Client.Call but
// in a much cheaper way. It assumes the underlying connection is not being
// shared with multiple concurrent RPCs. The request/response must be syncronous.
func CallWithCodec(cc rpc.ClientCodec, method string, args interface{}, resp interface{}) error {
	request := rpc.Request{
		Seq:           atomic.AddUint64(&nextCallSeq, 1),
		ServiceMethod: method,
	}
	if err := cc.WriteRequest(&request, args); err != nil {
		return err
	}
	var response rpc.Response
	if err := cc.ReadResponseHeader(&response); err != nil {
		return err
	}
	if response.Error != "" {
		err := errors.New(response.Error)
		if readErr := cc.ReadResponseBody(nil); readErr != nil {
			err = multierror.Append(err, readErr)
		}
		return rpc.ServerError(err.Error())
	}
	if err := cc.ReadResponseBody(resp); err != nil {
		return err
	}
	return nil
}
