// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package clientv3

import (
	"fmt"
	"sync"
	"time"

	v3rpc "github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	mvccpb "github.com/coreos/etcd/mvcc/mvccpb"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

const (
	EventTypeDelete = mvccpb.DELETE
	EventTypePut    = mvccpb.PUT

	closeSendErrTimeout = 250 * time.Millisecond
)

type Event mvccpb.Event

type WatchChan <-chan WatchResponse

type Watcher interface {
	// Watch watches on a key or prefix. The watched events will be returned
	// through the returned channel.
	// If the watch is slow or the required rev is compacted, the watch request
	// might be canceled from the server-side and the chan will be closed.
	// 'opts' can be: 'WithRev' and/or 'WithPrefix'.
	Watch(ctx context.Context, key string, opts ...OpOption) WatchChan

	// Close closes the watcher and cancels all watch requests.
	Close() error
}

type WatchResponse struct {
	Header pb.ResponseHeader
	Events []*Event

	// CompactRevision is the minimum revision the watcher may receive.
	CompactRevision int64

	// Canceled is used to indicate watch failure.
	// If the watch failed and the stream was about to close, before the channel is closed,
	// the channel sends a final response that has Canceled set to true with a non-nil Err().
	Canceled bool

	// created is used to indicate the creation of the watcher.
	created bool

	closeErr error
}

// IsCreate returns true if the event tells that the key is newly created.
func (e *Event) IsCreate() bool {
	return e.Type == EventTypePut && e.Kv.CreateRevision == e.Kv.ModRevision
}

// IsModify returns true if the event tells that a new value is put on existing key.
func (e *Event) IsModify() bool {
	return e.Type == EventTypePut && e.Kv.CreateRevision != e.Kv.ModRevision
}

// Err is the error value if this WatchResponse holds an error.
func (wr *WatchResponse) Err() error {
	switch {
	case wr.closeErr != nil:
		return v3rpc.Error(wr.closeErr)
	case wr.CompactRevision != 0:
		return v3rpc.ErrCompacted
	case wr.Canceled:
		return v3rpc.ErrFutureRev
	}
	return nil
}

// IsProgressNotify returns true if the WatchResponse is progress notification.
func (wr *WatchResponse) IsProgressNotify() bool {
	return len(wr.Events) == 0 && !wr.Canceled && !wr.created && wr.CompactRevision == 0 && wr.Header.Revision != 0
}

// watcher implements the Watcher interface
type watcher struct {
	remote pb.WatchClient

	// mu protects the grpc streams map
	mu sync.RWMutex
	// streams holds all the active grpc streams keyed by ctx value.
	streams map[string]*watchGrpcStream
}

// watchGrpcStream tracks all watch resources attached to a single grpc stream.
type watchGrpcStream struct {
	owner  *watcher
	remote pb.WatchClient

	// ctx controls internal remote.Watch requests
	ctx context.Context
	// ctxKey is the key used when looking up this stream's context
	ctxKey string
	cancel context.CancelFunc

	// substreams holds all active watchers on this grpc stream
	substreams map[int64]*watcherStream
	// resuming holds all resuming watchers on this grpc stream
	resuming []*watcherStream

	// reqc sends a watch request from Watch() to the main goroutine
	reqc chan *watchRequest
	// respc receives data from the watch client
	respc chan *pb.WatchResponse
	// stopc is sent to the main goroutine to stop all processing
	stopc chan struct{}
	// donec closes to broadcast shutdown
	donec chan struct{}
	// errc transmits errors from grpc Recv to the watch stream reconn logic
	errc chan error
	// closingc gets the watcherStream of closing watchers
	closingc chan *watcherStream

	// resumec closes to signal that all substreams should begin resuming
	resumec chan struct{}
	// closeErr is the error that closed the watch stream
	closeErr error
}

// watchRequest is issued by the subscriber to start a new watcher
type watchRequest struct {
	ctx context.Context
	key string
	end string
	rev int64
	// progressNotify is for progress updates.
	progressNotify bool
	// get the previous key-value pair before the event happens
	prevKV bool
	// retc receives a chan WatchResponse once the watcher is established
	retc chan chan WatchResponse
}

// watcherStream represents a registered watcher
type watcherStream struct {
	// initReq is the request that initiated this request
	initReq watchRequest

	// outc publishes watch responses to subscriber
	outc chan WatchResponse
	// recvc buffers watch responses before publishing
	recvc chan *WatchResponse
	// donec closes when the watcherStream goroutine stops.
	donec chan struct{}
	// closing is set to true when stream should be scheduled to shutdown.
	closing bool
	// id is the registered watch id on the grpc stream
	id int64

	// buf holds all events received from etcd but not yet consumed by the client
	buf []*WatchResponse
}

func NewWatcher(c *Client) Watcher {
	return &watcher{
		remote:  pb.NewWatchClient(c.conn),
		streams: make(map[string]*watchGrpcStream),
	}
}

// never closes
var valCtxCh = make(chan struct{})
var zeroTime = time.Unix(0, 0)

// ctx with only the values; never Done
type valCtx struct{ context.Context }

func (vc *valCtx) Deadline() (time.Time, bool) { return zeroTime, false }
func (vc *valCtx) Done() <-chan struct{}       { return valCtxCh }
func (vc *valCtx) Err() error                  { return nil }

func (w *watcher) newWatcherGrpcStream(inctx context.Context) *watchGrpcStream {
	ctx, cancel := context.WithCancel(&valCtx{inctx})
	wgs := &watchGrpcStream{
		owner:      w,
		remote:     w.remote,
		ctx:        ctx,
		ctxKey:     fmt.Sprintf("%v", inctx),
		cancel:     cancel,
		substreams: make(map[int64]*watcherStream),

		respc:    make(chan *pb.WatchResponse),
		reqc:     make(chan *watchRequest),
		stopc:    make(chan struct{}),
		donec:    make(chan struct{}),
		errc:     make(chan error, 1),
		closingc: make(chan *watcherStream),
		resumec:  make(chan struct{}),
	}
	go wgs.run()
	return wgs
}

// Watch posts a watch request to run() and waits for a new watcher channel
func (w *watcher) Watch(ctx context.Context, key string, opts ...OpOption) WatchChan {
	ow := opWatch(key, opts...)

	wr := &watchRequest{
		ctx:            ctx,
		key:            string(ow.key),
		end:            string(ow.end),
		rev:            ow.rev,
		progressNotify: ow.progressNotify,
		prevKV:         ow.prevKV,
		retc:           make(chan chan WatchResponse, 1),
	}

	ok := false
	ctxKey := fmt.Sprintf("%v", ctx)

	// find or allocate appropriate grpc watch stream
	w.mu.Lock()
	if w.streams == nil {
		// closed
		w.mu.Unlock()
		ch := make(chan WatchResponse)
		close(ch)
		return ch
	}
	wgs := w.streams[ctxKey]
	if wgs == nil {
		wgs = w.newWatcherGrpcStream(ctx)
		w.streams[ctxKey] = wgs
	}
	donec := wgs.donec
	reqc := wgs.reqc
	w.mu.Unlock()

	// couldn't create channel; return closed channel
	closeCh := make(chan WatchResponse, 1)

	// submit request
	select {
	case reqc <- wr:
		ok = true
	case <-wr.ctx.Done():
	case <-donec:
		if wgs.closeErr != nil {
			closeCh <- WatchResponse{closeErr: wgs.closeErr}
			break
		}
		// retry; may have dropped stream from no ctxs
		return w.Watch(ctx, key, opts...)
	}

	// receive channel
	if ok {
		select {
		case ret := <-wr.retc:
			return ret
		case <-ctx.Done():
		case <-donec:
			if wgs.closeErr != nil {
				closeCh <- WatchResponse{closeErr: wgs.closeErr}
				break
			}
			// retry; may have dropped stream from no ctxs
			return w.Watch(ctx, key, opts...)
		}
	}

	close(closeCh)
	return closeCh
}

func (w *watcher) Close() (err error) {
	w.mu.Lock()
	streams := w.streams
	w.streams = nil
	w.mu.Unlock()
	for _, wgs := range streams {
		if werr := wgs.Close(); werr != nil {
			err = werr
		}
	}
	return err
}

func (w *watchGrpcStream) Close() (err error) {
	close(w.stopc)
	<-w.donec
	select {
	case err = <-w.errc:
	default:
	}
	return toErr(w.ctx, err)
}

func (w *watcher) closeStream(wgs *watchGrpcStream) {
	w.mu.Lock()
	close(wgs.donec)
	wgs.cancel()
	if w.streams != nil {
		delete(w.streams, wgs.ctxKey)
	}
	w.mu.Unlock()
}

func (w *watchGrpcStream) addSubstream(resp *pb.WatchResponse, ws *watcherStream) {
	if resp.WatchId == -1 {
		// failed; no channel
		close(ws.recvc)
		return
	}
	ws.id = resp.WatchId
	w.substreams[ws.id] = ws
}

func (w *watchGrpcStream) sendCloseSubstream(ws *watcherStream, resp *WatchResponse) {
	select {
	case ws.outc <- *resp:
	case <-ws.initReq.ctx.Done():
	case <-time.After(closeSendErrTimeout):
	}
	close(ws.outc)
}

func (w *watchGrpcStream) closeSubstream(ws *watcherStream) {
	// send channel response in case stream was never established
	select {
	case ws.initReq.retc <- ws.outc:
	default:
	}
	// close subscriber's channel
	if closeErr := w.closeErr; closeErr != nil && ws.initReq.ctx.Err() == nil {
		go w.sendCloseSubstream(ws, &WatchResponse{closeErr: w.closeErr})
	} else {
		close(ws.outc)
	}
	if ws.id != -1 {
		delete(w.substreams, ws.id)
		return
	}
	for i := range w.resuming {
		if w.resuming[i] == ws {
			w.resuming[i] = nil
			return
		}
	}
}

// run is the root of the goroutines for managing a watcher client
func (w *watchGrpcStream) run() {
	var wc pb.Watch_WatchClient
	var closeErr error

	// substreams marked to close but goroutine still running; needed for
	// avoiding double-closing recvc on grpc stream teardown
	closing := make(map[*watcherStream]struct{})

	defer func() {
		w.closeErr = closeErr
		// shutdown substreams and resuming substreams
		for _, ws := range w.substreams {
			if _, ok := closing[ws]; !ok {
				close(ws.recvc)
			}
		}
		for _, ws := range w.resuming {
			if _, ok := closing[ws]; ws != nil && !ok {
				close(ws.recvc)
			}
		}
		w.joinSubstreams()
		for toClose := len(w.substreams) + len(w.resuming); toClose > 0; toClose-- {
			w.closeSubstream(<-w.closingc)
		}

		w.owner.closeStream(w)
	}()

	// start a stream with the etcd grpc server
	if wc, closeErr = w.newWatchClient(); closeErr != nil {
		return
	}

	cancelSet := make(map[int64]struct{})

	for {
		select {
		// Watch() requested
		case wreq := <-w.reqc:
			outc := make(chan WatchResponse, 1)
			ws := &watcherStream{
				initReq: *wreq,
				id:      -1,
				outc:    outc,
				// unbufffered so resumes won't cause repeat events
				recvc: make(chan *WatchResponse),
			}

			ws.donec = make(chan struct{})
			go w.serveSubstream(ws, w.resumec)

			// queue up for watcher creation/resume
			w.resuming = append(w.resuming, ws)
			if len(w.resuming) == 1 {
				// head of resume queue, can register a new watcher
				wc.Send(ws.initReq.toPB())
			}
		// New events from the watch client
		case pbresp := <-w.respc:
			switch {
			case pbresp.Created:
				// response to head of queue creation
				if ws := w.resuming[0]; ws != nil {
					w.addSubstream(pbresp, ws)
					w.dispatchEvent(pbresp)
					w.resuming[0] = nil
				}
				if ws := w.nextResume(); ws != nil {
					wc.Send(ws.initReq.toPB())
				}
			case pbresp.Canceled:
				delete(cancelSet, pbresp.WatchId)
				if ws, ok := w.substreams[pbresp.WatchId]; ok {
					// signal to stream goroutine to update closingc
					close(ws.recvc)
					closing[ws] = struct{}{}
				}
			default:
				// dispatch to appropriate watch stream
				if ok := w.dispatchEvent(pbresp); ok {
					break
				}
				// watch response on unexpected watch id; cancel id
				if _, ok := cancelSet[pbresp.WatchId]; ok {
					break
				}
				cancelSet[pbresp.WatchId] = struct{}{}
				cr := &pb.WatchRequest_CancelRequest{
					CancelRequest: &pb.WatchCancelRequest{
						WatchId: pbresp.WatchId,
					},
				}
				req := &pb.WatchRequest{RequestUnion: cr}
				wc.Send(req)
			}
		// watch client failed to recv; spawn another if possible
		case err := <-w.errc:
			if toErr(w.ctx, err) == v3rpc.ErrNoLeader {
				closeErr = err
				return
			}
			if wc, closeErr = w.newWatchClient(); closeErr != nil {
				return
			}
			if ws := w.nextResume(); ws != nil {
				wc.Send(ws.initReq.toPB())
			}
			cancelSet = make(map[int64]struct{})
		case <-w.stopc:
			return
		case ws := <-w.closingc:
			w.closeSubstream(ws)
			delete(closing, ws)
			if len(w.substreams)+len(w.resuming) == 0 {
				// no more watchers on this stream, shutdown
				return
			}
		}
	}
}

// nextResume chooses the next resuming to register with the grpc stream. Abandoned
// streams are marked as nil in the queue since the head must wait for its inflight registration.
func (w *watchGrpcStream) nextResume() *watcherStream {
	for len(w.resuming) != 0 {
		if w.resuming[0] != nil {
			return w.resuming[0]
		}
		w.resuming = w.resuming[1:len(w.resuming)]
	}
	return nil
}

// dispatchEvent sends a WatchResponse to the appropriate watcher stream
func (w *watchGrpcStream) dispatchEvent(pbresp *pb.WatchResponse) bool {
	ws, ok := w.substreams[pbresp.WatchId]
	if !ok {
		return false
	}
	events := make([]*Event, len(pbresp.Events))
	for i, ev := range pbresp.Events {
		events[i] = (*Event)(ev)
	}
	wr := &WatchResponse{
		Header:          *pbresp.Header,
		Events:          events,
		CompactRevision: pbresp.CompactRevision,
		created:         pbresp.Created,
		Canceled:        pbresp.Canceled,
	}
	select {
	case ws.recvc <- wr:
	case <-ws.donec:
		return false
	}
	return true
}

// serveWatchClient forwards messages from the grpc stream to run()
func (w *watchGrpcStream) serveWatchClient(wc pb.Watch_WatchClient) {
	for {
		resp, err := wc.Recv()
		if err != nil {
			select {
			case w.errc <- err:
			case <-w.donec:
			}
			return
		}
		select {
		case w.respc <- resp:
		case <-w.donec:
			return
		}
	}
}

// serveSubstream forwards watch responses from run() to the subscriber
func (w *watchGrpcStream) serveSubstream(ws *watcherStream, resumec chan struct{}) {
	if ws.closing {
		panic("created substream goroutine but substream is closing")
	}

	// nextRev is the minimum expected next revision
	nextRev := ws.initReq.rev
	resuming := false
	defer func() {
		if !resuming {
			ws.closing = true
		}
		close(ws.donec)
		if !resuming {
			w.closingc <- ws
		}
	}()

	emptyWr := &WatchResponse{}
	for {
		curWr := emptyWr
		outc := ws.outc

		if len(ws.buf) > 0 && ws.buf[0].created {
			select {
			case ws.initReq.retc <- ws.outc:
			default:
			}
			ws.buf = ws.buf[1:]
		}

		if len(ws.buf) > 0 {
			curWr = ws.buf[0]
		} else {
			outc = nil
		}
		select {
		case outc <- *curWr:
			if ws.buf[0].Err() != nil {
				return
			}
			ws.buf[0] = nil
			ws.buf = ws.buf[1:]
		case wr, ok := <-ws.recvc:
			if !ok {
				// shutdown from closeSubstream
				return
			}
			// TODO pause channel if buffer gets too large
			ws.buf = append(ws.buf, wr)
			nextRev = wr.Header.Revision
			if len(wr.Events) > 0 {
				nextRev = wr.Events[len(wr.Events)-1].Kv.ModRevision + 1
			}
			ws.initReq.rev = nextRev
		case <-ws.initReq.ctx.Done():
			return
		case <-resumec:
			resuming = true
			return
		}
	}
	// lazily send cancel message if events on missing id
}

func (w *watchGrpcStream) newWatchClient() (pb.Watch_WatchClient, error) {
	// connect to grpc stream
	wc, err := w.openWatchClient()
	if err != nil {
		return nil, v3rpc.Error(err)
	}
	// mark all substreams as resuming
	if len(w.substreams)+len(w.resuming) > 0 {
		close(w.resumec)
		w.resumec = make(chan struct{})
		w.joinSubstreams()
		for _, ws := range w.substreams {
			ws.id = -1
			w.resuming = append(w.resuming, ws)
		}
		for _, ws := range w.resuming {
			if ws == nil || ws.closing {
				continue
			}
			ws.donec = make(chan struct{})
			go w.serveSubstream(ws, w.resumec)
		}
	}
	w.substreams = make(map[int64]*watcherStream)
	// receive data from new grpc stream
	go w.serveWatchClient(wc)
	return wc, nil
}

// joinSubstream waits for all substream goroutines to complete
func (w *watchGrpcStream) joinSubstreams() {
	for _, ws := range w.substreams {
		<-ws.donec
	}
	for _, ws := range w.resuming {
		if ws != nil {
			<-ws.donec
		}
	}
}

// openWatchClient retries opening a watchclient until retryConnection fails
func (w *watchGrpcStream) openWatchClient() (ws pb.Watch_WatchClient, err error) {
	for {
		select {
		case <-w.stopc:
			if err == nil {
				return nil, context.Canceled
			}
			return nil, err
		default:
		}
		if ws, err = w.remote.Watch(w.ctx, grpc.FailFast(false)); ws != nil && err == nil {
			break
		}
		if isHaltErr(w.ctx, err) {
			return nil, v3rpc.Error(err)
		}
	}
	return ws, nil
}

// toPB converts an internal watch request structure to its protobuf messagefunc (wr *watchRequest)
func (wr *watchRequest) toPB() *pb.WatchRequest {
	req := &pb.WatchCreateRequest{
		StartRevision:  wr.rev,
		Key:            []byte(wr.key),
		RangeEnd:       []byte(wr.end),
		ProgressNotify: wr.progressNotify,
		PrevKv:         wr.prevKV,
	}
	cr := &pb.WatchRequest_CreateRequest{CreateRequest: req}
	return &pb.WatchRequest{RequestUnion: cr}
}
