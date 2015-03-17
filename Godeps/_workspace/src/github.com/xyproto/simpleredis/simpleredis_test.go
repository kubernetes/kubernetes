package simpleredis

import (
	"testing"
)

var pool *ConnectionPool

func TestLocalConnection(t *testing.T) {
	if err := TestConnection(); err != nil {
		t.Errorf(err.Error())
	}
}

func TestRemoteConnection(t *testing.T) {
	if err := TestConnectionHost("foobared@ :6379"); err != nil {
		t.Errorf(err.Error())
	}
}

func TestConnectionPool(t *testing.T) {
	pool = NewConnectionPool()
}

func TestConnectionPoolHost(t *testing.T) {
	pool = NewConnectionPoolHost("localhost:6379")
}

// Tests with password "foobared" if the previous connection test
// did not result in a connection that responds to PING.
func TestConnectionPoolHostPassword(t *testing.T) {
	if !pool.Ping() {
		// Try connecting with the default password
		pool = NewConnectionPoolHost("foobared@localhost:6379")
	}
}

func TestList(t *testing.T) {
	const (
		listname = "abc123_test_test_test_123abc"
		testdata = "123abc"
	)
	list := NewList(pool, listname)
	list.SelectDatabase(1)
	if err := list.Add(testdata); err != nil {
		t.Errorf("Error, could not add item to list! %s", err.Error())
	}
	items, err := list.GetAll()
	if len(items) != 1 {
		t.Errorf("Error, wrong list length! %v", len(items))
	}
	if (len(items) > 0) && (items[0] != testdata) {
		t.Errorf("Error, wrong list contents! %v", items)
	}
	err = list.Remove()
	if err != nil {
		t.Errorf("Error, could not remove list! %s", err.Error())
	}
}

func TestTwoFields(t *testing.T) {
	test, test23, ok := twoFields("test1@test2@test3", "@")
	if ok && ((test != "test1") || (test23 != "test2@test3")) {
		t.Error("Error in twoFields functions")
	}
}
