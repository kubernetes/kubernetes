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
	return len(wr.Events) == 0 && !wr.Canceled
}

// watcher implements the Watcher interface
type watcher struct {
	remote pb.WatchClient

	// mu protects the grpc streams map
	mu sync.RWMutex
	// streams holds all the active grpc streams keyed by ctx value.
	streams map[string]*watchGrpcStream
}

type watchGrpcStream struct {
	owner  *watcher
	remote pb.WatchClient

	// ctx controls internal remote.Watch requests
	ctx context.Context
	// ctxKey is the key used when looking up this stream's context
	ctxKey string
	cancel context.CancelFunc

	// mu protects the streams map
	mu sync.RWMutex
	// streams holds all active watchers
	streams map[int64]*watcherStream

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

	// the error that closed the watch stream
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
	// retc receives a chan WatchResponse once the watcher is established
	retc chan chan WatchResponse
}

// watcherStream represents a registered watcher
type watcherStream struct {
	// initReq is the request that initiated this request
	initReq watchRequest

	// outc publishes watch responses to subscriber
	outc chan<- WatchResponse
	// recvc buffers watch responses before publishing
	recvc chan *WatchResponse
	id    int64

	// lastRev is revision last successfully sent over outc
	lastRev int64
	// resumec indicates the stream must recover at a given revision
	resumec chan int64
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
		owner:   w,
		remote:  w.remote,
		ctx:     ctx,
		ctxKey:  fmt.Sprintf("%v", inctx),
		cancel:  cancel,
		streams: make(map[int64]*watcherStream),

		respc: make(chan *pb.WatchResponse),
		reqc:  make(chan *watchRequest),
		stopc: make(chan struct{}),
		donec: make(chan struct{}),
		errc:  make(chan error, 1),
	}
	go wgs.run()
	return wgs
}

// Watch posts a watch request to run() and waits for a new watcher channel
func (w *watcher) Watch(ctx context.Context, key string, opts ...OpOption) WatchChan {
	ow := opWatch(key, opts...)

	retc := make(chan chan WatchResponse, 1)
	wr := &watchRequest{
		ctx:            ctx,
		key:            string(ow.key),
		end:            string(ow.end),
		rev:            ow.rev,
		progressNotify: ow.progressNotify,
		retc:           retc,
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
		wgs.stopIfEmpty()
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
		case ret := <-retc:
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
	w.mu.Lock()
	if w.stopc != nil {
		close(w.stopc)
		w.stopc = nil
	}
	w.mu.Unlock()
	<-w.donec
	select {
	case err = <-w.errc:
	default:
	}
	return toErr(w.ctx, err)
}

func (w *watchGrpcStream) addStream(resp *pb.WatchResponse, pendingReq *watchRequest) {
	if pendingReq == nil {
		// no pending request; ignore
		return
	}
	if resp.Canceled || resp.CompactRevision != 0 {
		// a cancel at id creation time means the start revision has
		// been compacted out of the store
		ret := make(chan WatchResponse, 1)
		ret <- WatchResponse{
			Header:          *resp.Header,
			CompactRevision: resp.CompactRevision,
			Canceled:        true}
		close(ret)
		pendingReq.retc <- ret
		return
	}

	ret := make(chan WatchResponse)
	if resp.WatchId == -1 {
		// failed; no channel
		close(ret)
		pendingReq.retc <- ret
		return
	}

	ws := &watcherStream{
		initReq: *pendingReq,
		id:      resp.WatchId,
		outc:    ret,
		// buffered so unlikely to block on sending while holding mu
		recvc:   make(chan *WatchResponse, 4),
		resumec: make(chan int64),
	}

	if pendingReq.rev == 0 {
		// note the header revision so that a put following a current watcher
		// disconnect will arrive on the watcher channel after reconnect
		ws.initReq.rev = resp.Header.Revision
	}

	w.mu.Lock()
	w.streams[ws.id] = ws
	w.mu.Unlock()

	// pass back the subscriber channel for the watcher
	pendingReq.retc <- ret

	// send messages to subscriber
	go w.serveStream(ws)
}

// closeStream closes the watcher resources and removes it
func (w *watchGrpcStream) closeStream(ws *watcherStream) {
	w.mu.Lock()
	// cancels request stream; subscriber receives nil channel
	close(ws.initReq.retc)
	// close subscriber's channel
	close(ws.outc)
	delete(w.streams, ws.id)
	w.mu.Unlock()
}

// run is the root of the goroutines for managing a watcher client
func (w *watchGrpcStream) run() {
	var wc pb.Watch_WatchClient
	var closeErr error

	defer func() {
		w.owner.mu.Lock()
		w.closeErr = closeErr
		if w.owner.streams != nil {
			delete(w.owner.streams, w.ctxKey)
		}
		close(w.donec)
		w.owner.mu.Unlock()
		w.cancel()
	}()

	// already stopped?
	w.mu.RLock()
	stopc := w.stopc
	w.mu.RUnlock()
	if stopc == nil {
		return
	}

	// start a stream with the etcd grpc server
	if wc, closeErr = w.newWatchClient(); closeErr != nil {
		return
	}

	var pendingReq, failedReq *watchRequest
	curReqC := w.reqc
	cancelSet := make(map[int64]struct{})

	for {
		select {
		// Watch() requested
		case pendingReq = <-curReqC:
			// no more watch requests until there's a response
			curReqC = nil
			if err := wc.Send(pendingReq.toPB()); err == nil {
				// pendingReq now waits on w.respc
				break
			}
			failedReq = pendingReq
		// New events from the watch client
		case pbresp := <-w.respc:
			switch {
			case pbresp.Created:
				// response to pending req, try to add
				w.addStream(pbresp, pendingReq)
				pendingReq = nil
				curReqC = w.reqc
			case pbresp.Canceled:
				delete(cancelSet, pbresp.WatchId)
				// shutdown serveStream, if any
				w.mu.Lock()
				if ws, ok := w.streams[pbresp.WatchId]; ok {
					close(ws.recvc)
					delete(w.streams, ws.id)
				}
				numStreams := len(w.streams)
				w.mu.Unlock()
				if numStreams == 0 {
					// don't leak watcher streams
					return
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
		// TODO report watch client errors from errc?
		case err := <-w.errc:
			if toErr(w.ctx, err) == v3rpc.ErrNoLeader {
				closeErr = err
				return
			}
			if wc, closeErr = w.newWatchClient(); closeErr != nil {
				return
			}
			curReqC = w.reqc
			if pendingReq != nil {
				failedReq = pendingReq
			}
			cancelSet = make(map[int64]struct{})
		case <-stopc:
			return
		}

		// send failed; queue for retry
		if failedReq != nil {
			go func(wr *watchRequest) {
				select {
				case w.reqc <- wr:
				case <-wr.ctx.Done():
				case <-w.donec:
				}
			}(pendingReq)
			failedReq = nil
			pendingReq = nil
		}
	}
}

// dispatchEvent sends a WatchResponse to the appropriate watcher stream
func (w *watchGrpcStream) dispatchEvent(pbresp *pb.WatchResponse) bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	ws, ok := w.streams[pbresp.WatchId]
	events := make([]*Event, len(pbresp.Events))
	for i, ev := range pbresp.Events {
		events[i] = (*Event)(ev)
	}
	if ok {
		wr := &WatchResponse{
			Header:          *pbresp.Header,
			Events:          events,
			CompactRevision: pbresp.CompactRevision,
			Canceled:        pbresp.Canceled}
		ws.recvc <- wr
	}
	return ok
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

// serveStream forwards watch responses from run() to the subscriber
func (w *watchGrpcStream) serveStream(ws *watcherStream) {
	var closeErr error
	emptyWr := &WatchResponse{}
	wrs := []*WatchResponse{}
	resuming := false
	closing := false
	for !closing {
		curWr := emptyWr
		outc := ws.outc
		if len(wrs) > 0 {
			curWr = wrs[0]
		} else {
			outc = nil
		}
		select {
		case outc <- *curWr:
			if wrs[0].Err() != nil {
				closing = true
				break
			}
			var newRev int64
			if len(wrs[0].Events) > 0 {
				newRev = wrs[0].Events[len(wrs[0].Events)-1].Kv.ModRevision
			} else {
				newRev = wrs[0].Header.Revision
			}
			if newRev != ws.lastRev {
				ws.lastRev = newRev
			}
			wrs[0] = nil
			wrs = wrs[1:]
		case wr, ok := <-ws.recvc:
			if !ok {
				// shutdown from closeStream
				return
			}
			// resume up to last seen event if disconnected
			if resuming && wr.Err() == nil {
				resuming = false
				// trim events already seen
				for i := 0; i < len(wr.Events); i++ {
					if wr.Events[i].Kv.ModRevision > ws.lastRev {
						wr.Events = wr.Events[i:]
						break
					}
				}
				// only forward new events
				if wr.Events[0].Kv.ModRevision == ws.lastRev {
					break
				}
			}
			resuming = false
			// TODO don't keep buffering if subscriber stops reading
			wrs = append(wrs, wr)
		case resumeRev := <-ws.resumec:
			wrs = nil
			resuming = true
			if resumeRev == -1 {
				// pause serving stream while resume gets set up
				break
			}
			if resumeRev != ws.lastRev {
				panic("unexpected resume revision")
			}
		case <-w.donec:
			closing = true
			closeErr = w.closeErr
		case <-ws.initReq.ctx.Done():
			closing = true
		}
	}

	// try to send off close error
	if closeErr != nil {
		select {
		case ws.outc <- WatchResponse{closeErr: w.closeErr}:
		case <-w.donec:
		case <-time.After(closeSendErrTimeout):
		}
	}

	w.closeStream(ws)
	w.stopIfEmpty()
	// lazily send cancel message if events on missing id
}

func (wgs *watchGrpcStream) stopIfEmpty() {
	wgs.mu.Lock()
	if len(wgs.streams) == 0 && wgs.stopc != nil {
		close(wgs.stopc)
		wgs.stopc = nil
	}
	wgs.mu.Unlock()
}

func (w *watchGrpcStream) newWatchClient() (pb.Watch_WatchClient, error) {
	ws, rerr := w.resume()
	if rerr != nil {
		return nil, rerr
	}
	go w.serveWatchClient(ws)
	return ws, nil
}

// resume creates a new WatchClient with all current watchers reestablished
func (w *watchGrpcStream) resume() (ws pb.Watch_WatchClient, err error) {
	for {
		if ws, err = w.openWatchClient(); err != nil {
			break
		} else if err = w.resumeWatchers(ws); err == nil {
			break
		}
	}
	return ws, v3rpc.Error(err)
}

// openWatchClient retries opening a watchclient until retryConnection fails
func (w *watchGrpcStream) openWatchClient() (ws pb.Watch_WatchClient, err error) {
	for {
		w.mu.Lock()
		stopc := w.stopc
		w.mu.Unlock()
		if stopc == nil {
			if err == nil {
				err = context.Canceled
			}
			return nil, err
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

// resumeWatchers rebuilds every registered watcher on a new client
func (w *watchGrpcStream) resumeWatchers(wc pb.Watch_WatchClient) error {
	w.mu.RLock()
	streams := make([]*watcherStream, 0, len(w.streams))
	for _, ws := range w.streams {
		streams = append(streams, ws)
	}
	w.mu.RUnlock()

	for _, ws := range streams {
		// pause serveStream
		ws.resumec <- -1

		// reconstruct watcher from initial request
		if ws.lastRev != 0 {
			ws.initReq.rev = ws.lastRev
		}
		if err := wc.Send(ws.initReq.toPB()); err != nil {
			return err
		}

		// wait for request ack
		resp, err := wc.Recv()
		if err != nil {
			return err
		} else if len(resp.Events) != 0 || !resp.Created {
			return fmt.Errorf("watcher: unexpected response (%+v)", resp)
		}

		// id may be different since new remote watcher; update map
		w.mu.Lock()
		delete(w.streams, ws.id)
		ws.id = resp.WatchId
		w.streams[ws.id] = ws
		w.mu.Unlock()

		// unpause serveStream
		ws.resumec <- ws.lastRev
	}
	return nil
}

// toPB converts an internal watch request structure to its protobuf messagefunc (wr *watchRequest)
func (wr *watchRequest) toPB() *pb.WatchRequest {
	req := &pb.WatchCreateRequest{
		StartRevision:  wr.rev,
		Key:            []byte(wr.key),
		RangeEnd:       []byte(wr.end),
		ProgressNotify: wr.progressNotify,
	}
	cr := &pb.WatchRequest_CreateRequest{CreateRequest: req}
	return &pb.WatchRequest{RequestUnion: cr}
}
