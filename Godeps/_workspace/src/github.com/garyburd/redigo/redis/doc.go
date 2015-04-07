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

// Package redis is a client for the Redis database.
//
// The Redigo FAQ (https://github.com/garyburd/redigo/wiki/FAQ) contains more
// documentation about this package.
//
// Connections
//
// The Conn interface is the primary interface for working with Redis.
// Applications create connections by calling the Dial, DialWithTimeout or
// NewConn functions. In the future, functions will be added for creating
// sharded and other types of connections.
//
// The application must call the connection Close method when the application
// is done with the connection.
//
// Executing Commands
//
// The Conn interface has a generic method for executing Redis commands:
//
//  Do(commandName string, args ...interface{}) (reply interface{}, err error)
//
// The Redis command reference (http://redis.io/commands) lists the available
// commands. An example of using the Redis APPEND command is:
//
//  n, err := conn.Do("APPEND", "key", "value")
//
// The Do method converts command arguments to binary strings for transmission
// to the server as follows:
//
//  Go Type                 Conversion
//  []byte                  Sent as is
//  string                  Sent as is
//  int, int64              strconv.FormatInt(v)
//  float64                 strconv.FormatFloat(v, 'g', -1, 64)
//  bool                    true -> "1", false -> "0"
//  nil                     ""
//  all other types         fmt.Print(v)
//
// Redis command reply types are represented using the following Go types:
//
//  Redis type              Go type
//  error                   redis.Error
//  integer                 int64
//  simple string           string
//  bulk string             []byte or nil if value not present.
//  array                   []interface{} or nil if value not present.
//
// Use type assertions or the reply helper functions to convert from
// interface{} to the specific Go type for the command result.
//
// Pipelining
//
// Connections support pipelining using the Send, Flush and Receive methods.
//
//  Send(commandName string, args ...interface{}) error
//  Flush() error
//  Receive() (reply interface{}, err error)
//
// Send writes the command to the connection's output buffer. Flush flushes the
// connection's output buffer to the server. Receive reads a single reply from
// the server. The following example shows a simple pipeline.
//
//  c.Send("SET", "foo", "bar")
//  c.Send("GET", "foo")
//  c.Flush()
//  c.Receive() // reply from SET
//  v, err = c.Receive() // reply from GET
//
// The Do method combines the functionality of the Send, Flush and Receive
// methods. The Do method starts by writing the command and flushing the output
// buffer. Next, the Do method receives all pending replies including the reply
// for the command just sent by Do. If any of the received replies is an error,
// then Do returns the error. If there are no errors, then Do returns the last
// reply. If the command argument to the Do method is "", then the Do method
// will flush the output buffer and receive pending replies without sending a
// command.
//
// Use the Send and Do methods to implement pipelined transactions.
//
//  c.Send("MULTI")
//  c.Send("INCR", "foo")
//  c.Send("INCR", "bar")
//  r, err := c.Do("EXEC")
//  fmt.Println(r) // prints [1, 1]
//
// Concurrency
//
// Connections do not support concurrent calls to the write methods (Send,
// Flush) or concurrent calls to the read method (Receive). Connections do
// allow a concurrent reader and writer.
//
// Because the Do method combines the functionality of Send, Flush and Receive,
// the Do method cannot be called concurrently with the other methods.
//
// For full concurrent access to Redis, use the thread-safe Pool to get and
// release connections from within a goroutine.
//
// Publish and Subscribe
//
// Use the Send, Flush and Receive methods to implement Pub/Sub subscribers.
//
//  c.Send("SUBSCRIBE", "example")
//  c.Flush()
//  for {
//      reply, err := c.Receive()
//      if err != nil {
//          return err
//      }
//      // process pushed message
//  }
//
// The PubSubConn type wraps a Conn with convenience methods for implementing
// subscribers. The Subscribe, PSubscribe, Unsubscribe and PUnsubscribe methods
// send and flush a subscription management command. The receive method
// converts a pushed message to convenient types for use in a type switch.
//
//  psc := redis.PubSubConn{c}
//  psc.Subscribe("example")
//  for {
//      switch v := psc.Receive().(type) {
//      case redis.Message:
//          fmt.Printf("%s: message: %s\n", v.Channel, v.Data)
//      case redis.Subscription:
//          fmt.Printf("%s: %s %d\n", v.Channel, v.Kind, v.Count)
//      case error:
//          return v
//      }
//  }
//
// Reply Helpers
//
// The Bool, Int, Bytes, String, Strings and Values functions convert a reply
// to a value of a specific type. To allow convenient wrapping of calls to the
// connection Do and Receive methods, the functions take a second argument of
// type error.  If the error is non-nil, then the helper function returns the
// error. If the error is nil, the function converts the reply to the specified
// type:
//
//  exists, err := redis.Bool(c.Do("EXISTS", "foo"))
//  if err != nil {
//      // handle error return from c.Do or type conversion error.
//  }
//
// The Scan function converts elements of a array reply to Go types:
//
//  var value1 int
//  var value2 string
//  reply, err := redis.Values(c.Do("MGET", "key1", "key2"))
//  if err != nil {
//      // handle error
//  }
//   if _, err := redis.Scan(reply, &value1, &value2); err != nil {
//      // handle error
//  }
package redis // import "github.com/garyburd/redigo/redis"
