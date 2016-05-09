Simple Redis
============

[![Build Status](https://travis-ci.org/xyproto/simpleredis.svg?branch=master)](https://travis-ci.org/xyproto/simpleredis)
[![GoDoc](https://godoc.org/github.com/xyproto/simpleredis?status.svg)](http://godoc.org/github.com/xyproto/simpleredis)


Easy way to use Redis from Go.


Online API Documentation
------------------------

[godoc.org](http://godoc.org/github.com/xyproto/simpleredis)


Features and limitations
------------------------

* Supports simple use of lists, hashmaps, sets and key/values
* Deals mainly with strings
* Uses the [redigo](https://github.com/garyburd/redigo) package


Example usage
-------------

~~~go
package main

import (
	"log"

	"github.com/xyproto/simpleredis"
)

func main() {
	// Check if the redis service is up
	if err := simpleredis.TestConnection(); err != nil {
		log.Fatalln("Could not connect to Redis. Is the service up and running?")
	}

	// Use instead for testing if a different host/port is up.
	// simpleredis.TestConnectionHost("localhost:6379")

	// Create a connection pool, connect to the given redis server
	pool := simpleredis.NewConnectionPool()

	// Use this for connecting to a different redis host/port
	// pool := simpleredis.NewConnectionPoolHost("localhost:6379")

	// For connecting to a different redis host/port, with a password
	// pool := simpleredis.NewConnectionPoolHost("password@redishost:6379")

	// Close the connection pool right after this function returns
	defer pool.Close()

	// Create a list named "greetings"
	list := simpleredis.NewList(pool, "greetings")

	// Add "hello" to the list, check if there are errors
	if list.Add("hello") != nil {
		log.Fatalln("Could not add an item to list!")
	}

	// Get the last item of the list
	if item, err := list.GetLast(); err != nil {
		log.Fatalln("Could not fetch the last item from the list!")
	} else {
		log.Println("The value of the stored item is:", item)
	}

	// Remove the list
	if list.Remove() != nil {
		log.Fatalln("Could not remove the list!")
	}
}
~~~

Testing
-------

Redis must be up and running locally for the `go test` tests to work.


Version, license and author
---------------------------

* Version: 1.0
* License: MIT
* Author: Alexander F Rødseth

