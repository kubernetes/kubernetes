// Copyright 2012 Gary Burd
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package redis_test

import (
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/garyburd/redigo/redis"
)

func publish(channel, value interface{}) {
	c, err := dial()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer c.Close()
	c.Do("PUBLISH", channel, value)
}

// Applications can receive pushed messages from one goroutine and manage subscriptions from another goroutine.
func ExamplePubSubConn() {
	c, err := dial()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer c.Close()
	var wg sync.WaitGroup
	wg.Add(2)

	psc := redis.PubSubConn{Conn: c}

	// This goroutine receives and prints pushed notifications from the server.
	// The goroutine exits when the connection is unsubscribed from all
	// channels or there is an error.
	go func() {
		defer wg.Done()
		for {
			switch n := psc.Receive().(type) {
			case redis.Message:
				fmt.Printf("Message: %s %s\n", n.Channel, n.Data)
			case redis.PMessage:
				fmt.Printf("PMessage: %s %s %s\n", n.Pattern, n.Channel, n.Data)
			case redis.Subscription:
				fmt.Printf("Subscription: %s %s %d\n", n.Kind, n.Channel, n.Count)
				if n.Count == 0 {
					return
				}
			case error:
				fmt.Printf("error: %v\n", n)
				return
			}
		}
	}()

	// This goroutine manages subscriptions for the connection.
	go func() {
		defer wg.Done()

		psc.Subscribe("example")
		psc.PSubscribe("p*")

		// The following function calls publish a message using another
		// connection to the Redis server.
		publish("example", "hello")
		publish("example", "world")
		publish("pexample", "foo")
		publish("pexample", "bar")

		// Unsubscribe from all connections. This will cause the receiving
		// goroutine to exit.
		psc.Unsubscribe()
		psc.PUnsubscribe()
	}()

	wg.Wait()

	// Output:
	// Subscription: subscribe example 1
	// Subscription: psubscribe p* 2
	// Message: example hello
	// Message: example world
	// PMessage: p* pexample foo
	// PMessage: p* pexample bar
	// Subscription: unsubscribe example 1
	// Subscription: punsubscribe p* 0
}

func expectPushed(t *testing.T, c redis.PubSubConn, message string, expected interface{}) {
	actual := c.Receive()
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("%s = %v, want %v", message, actual, expected)
	}
}

func TestPushed(t *testing.T) {
	pc, err := redis.DialDefaultServer()
	if err != nil {
		t.Fatalf("error connection to database, %v", err)
	}
	defer pc.Close()

	sc, err := redis.DialDefaultServer()
	if err != nil {
		t.Fatalf("error connection to database, %v", err)
	}
	defer sc.Close()

	c := redis.PubSubConn{Conn: sc}

	c.Subscribe("c1")
	expectPushed(t, c, "Subscribe(c1)", redis.Subscription{Kind: "subscribe", Channel: "c1", Count: 1})
	c.Subscribe("c2")
	expectPushed(t, c, "Subscribe(c2)", redis.Subscription{Kind: "subscribe", Channel: "c2", Count: 2})
	c.PSubscribe("p1")
	expectPushed(t, c, "PSubscribe(p1)", redis.Subscription{Kind: "psubscribe", Channel: "p1", Count: 3})
	c.PSubscribe("p2")
	expectPushed(t, c, "PSubscribe(p2)", redis.Subscription{Kind: "psubscribe", Channel: "p2", Count: 4})
	c.PUnsubscribe()
	expectPushed(t, c, "Punsubscribe(p1)", redis.Subscription{Kind: "punsubscribe", Channel: "p1", Count: 3})
	expectPushed(t, c, "Punsubscribe()", redis.Subscription{Kind: "punsubscribe", Channel: "p2", Count: 2})

	pc.Do("PUBLISH", "c1", "hello")
	expectPushed(t, c, "PUBLISH c1 hello", redis.Message{Channel: "c1", Data: []byte("hello")})

	c.Ping("hello")
	expectPushed(t, c, `Ping("hello")`, redis.Pong{Data: "hello"})

	c.Conn.Send("PING")
	c.Conn.Flush()
	expectPushed(t, c, `Send("PING")`, redis.Pong{})
}
