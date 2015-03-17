// Copyright 2013 Gary Burd
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
	"github.com/garyburd/redigo/redis"
)

// zpop pops a value from the ZSET key using WATCH/MULTI/EXEC commands.
func zpop(c redis.Conn, key string) (result string, err error) {

	defer func() {
		// Return connection to normal state on error.
		if err != nil {
			c.Do("DISCARD")
		}
	}()

	// Loop until transaction is successful.
	for {
		if _, err := c.Do("WATCH", key); err != nil {
			return "", err
		}

		members, err := redis.Strings(c.Do("ZRANGE", key, 0, 0))
		if err != nil {
			return "", err
		}
		if len(members) != 1 {
			return "", redis.ErrNil
		}

		c.Send("MULTI")
		c.Send("ZREM", key, members[0])
		queued, err := c.Do("EXEC")
		if err != nil {
			return "", err
		}

		if queued != nil {
			result = members[0]
			break
		}
	}

	return result, nil
}

// zpopScript pops a value from a ZSET.
var zpopScript = redis.NewScript(1, `
    local r = redis.call('ZRANGE', KEYS[1], 0, 0)
    if r ~= nil then
        r = r[1]
        redis.call('ZREM', KEYS[1], r)
    end
    return r
`)

// This example implements ZPOP as described at
// http://redis.io/topics/transactions using WATCH/MULTI/EXEC and scripting.
func Example_zpop() {
	c, err := dial()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer c.Close()

	// Add test data using a pipeline.

	for i, member := range []string{"red", "blue", "green"} {
		c.Send("ZADD", "zset", i, member)
	}
	if _, err := c.Do(""); err != nil {
		fmt.Println(err)
		return
	}

	// Pop using WATCH/MULTI/EXEC

	v, err := zpop(c, "zset")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(v)

	// Pop using a script.

	v, err = redis.String(zpopScript.Do(c, "zset"))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(v)

	// Output:
	// red
	// blue
}
