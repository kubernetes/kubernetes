package storageos

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"

	"github.com/storageos/go-api/types"
)

var (

	// EventAPIPrefix is a partial path to the HTTP endpoint.
	EventAPIPrefix = "event"

	// ErrNoSuchEvent is the error returned when the event does not exist.
	ErrNoSuchEvent = errors.New("no such event")
)

// EventList returns the list of available events.
func (c *Client) EventList(opts types.ListOptions) ([]*types.Event, error) {
	listOpts := doOptions{
		fieldSelector: opts.FieldSelector,
		labelSelector: opts.LabelSelector,
		context:       opts.Context,
	}
	resp, err := c.do("GET", EventAPIPrefix, listOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var events []*types.Event
	if err := json.NewDecoder(resp.Body).Decode(&events); err != nil {
		return nil, err
	}
	return events, nil
}

// Events returns a stream of events in the daemon. It's up to the caller to close the stream
// by cancelling the context. Once the stream has been completely read an io.EOF error will
// be sent over the error channel. If an error is sent all processing will be stopped. It's up
// to the caller to reopen the stream in the event of an error by reinvoking this method.
func (c *Client) Events(ctx context.Context, opts types.ListOptions) (<-chan types.Request, <-chan error) {

	// listOpts := doOptions{
	// 	fieldSelector: opts.FieldSelector,
	// 	labelSelector: opts.LabelSelector,
	// 	context:       ctx,
	// }

	messages := make(chan types.Request)
	errs := make(chan error, 1)

	// started := make(chan struct{})
	ws, _, err := websocket.DefaultDialer.Dial("ws://10.245.103.2:8000/v1/ws/event", nil)
	if err != nil {
		// close(started)
		// errs <- err
		log.Fatal(err)
	}
	// defer ws.Close()

	done := make(chan struct{})
	go func() {
		defer ws.Close()
		defer close(done)
		for {
			_, message, err := ws.ReadMessage()
			if err != nil {
				log.Println("read:", err)
				errs <- err
				return
			}
			// log.Printf("recv: %s", message)
			var request types.Request
			if err := json.Unmarshal(message, &request); err != nil {
				log.Printf("decode error: %s", message)
				errs <- err
				return
			}
			messages <- request
		}
	}()

	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	go func() {
		for {
			select {
			case t := <-ticker.C:
				log.Printf("tick: %s\n", t.String())
				err := ws.WriteMessage(websocket.TextMessage, []byte(t.String()))
				if err != nil {
					log.Println("write:", err)
					return
				}
			case <-ctx.Done():
				log.Println("done")
				err := ws.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
				if err != nil {
					log.Println("write close:", err)
					return
				}
				errs <- ctx.Err()
				select {
				case <-done:
				case <-time.After(time.Second):
				}
				ws.Close()
				return
			}
		}
	}()

	// go func() {
	// 	defer ws.Close()
	// 	defer close(errs)
	//
	// 	// query, err := buildEventsQueryParams(cli.version, options)
	// 	// if err != nil {
	// 	// 	close(started)
	// 	// 	errs <- err
	// 	// 	return
	// 	// }
	//
	// 	// resp, err := cli.get(ctx, "/events", query, nil)
	//
	// 	// decoder := json.NewDecoder(resp.Body)
	//
	// 	close(started)
	// 	for {
	// 		select {
	// 		case <-ctx.Done():
	// 			log.Println("done")
	// 			errs <- ctx.Err()
	// 			return
	// 		default:
	// 			log.Println("default")
	// 			_, message, err := ws.ReadMessage()
	// 			if err != nil {
	// 				log.Println("read:", err)
	// 				return
	// 			}
	// 			log.Printf("recv: %s", message)
	// 			var event types.Event
	// 			if err := json.Unmarshal(message, &event); err != nil {
	// 				log.Printf("decode error: %s", message)
	// 				errs <- err
	// 				return
	// 			}
	// 			log.Printf("sent: %v", event)
	// 			messages <- event
	//
	// 			// select {
	// 			// case messages <- event:
	// 			// case <-ctx.Done():
	// 			// 	errs <- ctx.Err()
	// 			// 	return
	// 			// }
	// 		}
	// 	}
	// }()
	// <-started
	log.Println("returning")
	return messages, errs
}

// Event returns a event by its reference.
func (c *Client) Event(ref string) (*types.Event, error) {
	resp, err := c.do("GET", EventAPIPrefix+"/"+ref, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, ErrNoSuchEvent
		}
		return nil, err
	}
	defer resp.Body.Close()
	var event types.Event
	if err := json.NewDecoder(resp.Body).Decode(&event); err != nil {
		return nil, err
	}
	return &event, nil
}
