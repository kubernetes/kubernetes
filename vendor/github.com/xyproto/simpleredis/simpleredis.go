// Easy way to use Redis from Go.
package simpleredis

import (
	"errors"
	"strconv"
	"strings"

	"github.com/garyburd/redigo/redis"
)

// Common for each of the redis datastructures used here
type redisDatastructure struct {
	pool    *ConnectionPool
	id      string
	dbindex int
}

type (
	// A pool of readily available Redis connections
	ConnectionPool redis.Pool

	List     redisDatastructure
	Set      redisDatastructure
	HashMap  redisDatastructure
	KeyValue redisDatastructure
)

const (
	// Version number. Stable API within major version numbers.
	Version = 1.0
	// The default [url]:port that Redis is running at
	defaultRedisServer = ":6379"
)

var (
	// How many connections should stay ready for requests, at a maximum?
	// When an idle connection is used, new idle connections are created.
	maxIdleConnections = 3
)

/* --- Helper functions --- */

// Connect to the local instance of Redis at port 6379
func newRedisConnection() (redis.Conn, error) {
	return newRedisConnectionTo(defaultRedisServer)
}

// Connect to host:port, host may be omitted, so ":6379" is valid.
// Will not try to AUTH with any given password (password@host:port).
func newRedisConnectionTo(hostColonPort string) (redis.Conn, error) {
	// Discard the password, if provided
	if _, theRest, ok := twoFields(hostColonPort, "@"); ok {
		hostColonPort = theRest
	}
	hostColonPort = strings.TrimSpace(hostColonPort)
	return redis.Dial("tcp", hostColonPort)
}

// Get a string from a list of results at a given position
func getString(bi []interface{}, i int) string {
	return string(bi[i].([]uint8))
}

// Test if the local Redis server is up and running
func TestConnection() (err error) {
	return TestConnectionHost(defaultRedisServer)
}

// Test if a given Redis server at host:port is up and running.
// Does not try to PING or AUTH.
func TestConnectionHost(hostColonPort string) (err error) {
	// Connect to the given host:port
	conn, err := newRedisConnectionTo(hostColonPort)
	if conn != nil {
		conn.Close()
	}
	defer func() {
		if r := recover(); r != nil {
			err = errors.New("Could not connect to redis server: " + hostColonPort)
		}
	}()
	return err
}

/* --- ConnectionPool functions --- */

// Create a new connection pool
func NewConnectionPool() *ConnectionPool {
	// The second argument is the maximum number of idle connections
	redisPool := redis.NewPool(newRedisConnection, maxIdleConnections)
	pool := ConnectionPool(*redisPool)
	return &pool
}

// Split a string into two parts, given a delimiter.
// Returns the two parts and true if it works out.
func twoFields(s, delim string) (string, string, bool) {
	if strings.Count(s, delim) != 1 {
		return s, "", false
	}
	fields := strings.Split(s, delim)
	return fields[0], fields[1], true
}

// Create a new connection pool given a host:port string.
// A password may be supplied as well, on the form "password@host:port".
func NewConnectionPoolHost(hostColonPort string) *ConnectionPool {
	// Create a redis Pool
	redisPool := redis.NewPool(
		// Anonymous function for calling new RedisConnectionTo with the host:port
		func() (redis.Conn, error) {
			conn, err := newRedisConnectionTo(hostColonPort)
			if err != nil {
				return nil, err
			}
			// If a password is given, use it to authenticate
			if password, _, ok := twoFields(hostColonPort, "@"); ok {
				if password != "" {
					if _, err := conn.Do("AUTH", password); err != nil {
						conn.Close()
						return nil, err
					}
				}
			}
			return conn, err
		},
		// Maximum number of idle connections to the redis database
		maxIdleConnections)
	pool := ConnectionPool(*redisPool)
	return &pool
}

// Set the number of maximum *idle* connections standing ready when
// creating new connection pools. When an idle connection is used,
// a new idle connection is created. The default is 3 and should be fine
// for most cases.
func SetMaxIdleConnections(maximum int) {
	maxIdleConnections = maximum
}

// Get one of the available connections from the connection pool, given a database index
func (pool *ConnectionPool) Get(dbindex int) redis.Conn {
	redisPool := redis.Pool(*pool)
	conn := redisPool.Get()
	// The default database index is 0
	if dbindex != 0 {
		// SELECT is not critical, ignore the return values
		conn.Do("SELECT", strconv.Itoa(dbindex))
	}
	return conn
}

// Ping the server by sending a PING command
func (pool *ConnectionPool) Ping() (pong bool) {
	redisPool := redis.Pool(*pool)
	conn := redisPool.Get()
	_, err := conn.Do("PING")
	return err == nil
}

// Close down the connection pool
func (pool *ConnectionPool) Close() {
	redisPool := redis.Pool(*pool)
	redisPool.Close()
}

/* --- List functions --- */

// Create a new list
func NewList(pool *ConnectionPool, id string) *List {
	return &List{pool, id, 0}
}

// Select a different database
func (rl *List) SelectDatabase(dbindex int) {
	rl.dbindex = dbindex
}

// Add an element to the list
func (rl *List) Add(value string) error {
	conn := rl.pool.Get(rl.dbindex)
	_, err := conn.Do("RPUSH", rl.id, value)
	return err
}

// Get all elements of a list
func (rl *List) GetAll() ([]string, error) {
	conn := rl.pool.Get(rl.dbindex)
	result, err := redis.Values(conn.Do("LRANGE", rl.id, "0", "-1"))
	strs := make([]string, len(result))
	for i := 0; i < len(result); i++ {
		strs[i] = getString(result, i)
	}
	return strs, err
}

// Get the last element of a list
func (rl *List) GetLast() (string, error) {
	conn := rl.pool.Get(rl.dbindex)
	result, err := redis.Values(conn.Do("LRANGE", rl.id, "-1", "-1"))
	if len(result) == 1 {
		return getString(result, 0), err
	}
	return "", err
}

// Get the last N elements of a list
func (rl *List) GetLastN(n int) ([]string, error) {
	conn := rl.pool.Get(rl.dbindex)
	result, err := redis.Values(conn.Do("LRANGE", rl.id, "-"+strconv.Itoa(n), "-1"))
	strs := make([]string, len(result))
	for i := 0; i < len(result); i++ {
		strs[i] = getString(result, i)
	}
	return strs, err
}

// Remove this list
func (rl *List) Remove() error {
	conn := rl.pool.Get(rl.dbindex)
	_, err := conn.Do("DEL", rl.id)
	return err
}

/* --- Set functions --- */

// Create a new set
func NewSet(pool *ConnectionPool, id string) *Set {
	return &Set{pool, id, 0}
}

// Select a different database
func (rs *Set) SelectDatabase(dbindex int) {
	rs.dbindex = dbindex
}

// Add an element to the set
func (rs *Set) Add(value string) error {
	conn := rs.pool.Get(rs.dbindex)
	_, err := conn.Do("SADD", rs.id, value)
	return err
}

// Check if a given value is in the set
func (rs *Set) Has(value string) (bool, error) {
	conn := rs.pool.Get(rs.dbindex)
	retval, err := conn.Do("SISMEMBER", rs.id, value)
	if err != nil {
		panic(err)
	}
	return redis.Bool(retval, err)
}

// Get all elements of the set
func (rs *Set) GetAll() ([]string, error) {
	conn := rs.pool.Get(rs.dbindex)
	result, err := redis.Values(conn.Do("SMEMBERS", rs.id))
	strs := make([]string, len(result))
	for i := 0; i < len(result); i++ {
		strs[i] = getString(result, i)
	}
	return strs, err
}

// Remove an element from the set
func (rs *Set) Del(value string) error {
	conn := rs.pool.Get(rs.dbindex)
	_, err := conn.Do("SREM", rs.id, value)
	return err
}

// Remove this set
func (rs *Set) Remove() error {
	conn := rs.pool.Get(rs.dbindex)
	_, err := conn.Do("DEL", rs.id)
	return err
}

/* --- HashMap functions --- */

// Create a new hashmap
func NewHashMap(pool *ConnectionPool, id string) *HashMap {
	return &HashMap{pool, id, 0}
}

// Select a different database
func (rh *HashMap) SelectDatabase(dbindex int) {
	rh.dbindex = dbindex
}

// Set a value in a hashmap given the element id (for instance a user id) and the key (for instance "password")
func (rh *HashMap) Set(elementid, key, value string) error {
	conn := rh.pool.Get(rh.dbindex)
	_, err := conn.Do("HSET", rh.id+":"+elementid, key, value)
	return err
}

// Get a value from a hashmap given the element id (for instance a user id) and the key (for instance "password")
func (rh *HashMap) Get(elementid, key string) (string, error) {
	conn := rh.pool.Get(rh.dbindex)
	result, err := redis.String(conn.Do("HGET", rh.id+":"+elementid, key))
	if err != nil {
		return "", err
	}
	return result, nil
}

// Check if a given elementid + key is in the hash map
func (rh *HashMap) Has(elementid, key string) (bool, error) {
	conn := rh.pool.Get(rh.dbindex)
	retval, err := conn.Do("HEXISTS", rh.id+":"+elementid, key)
	if err != nil {
		panic(err)
	}
	return redis.Bool(retval, err)
}

// Check if a given elementid exists as a hash map at all
func (rh *HashMap) Exists(elementid string) (bool, error) {
	// TODO: key is not meant to be a wildcard, check for "*"
	return hasKey(rh.pool, rh.id+":"+elementid, rh.dbindex)
}

// Get all elementid's for all hash elements
func (rh *HashMap) GetAll() ([]string, error) {
	conn := rh.pool.Get(rh.dbindex)
	result, err := redis.Values(conn.Do("KEYS", rh.id+":*"))
	strs := make([]string, len(result))
	idlen := len(rh.id)
	for i := 0; i < len(result); i++ {
		strs[i] = getString(result, i)[idlen+1:]
	}
	return strs, err
}

// Remove a key for an entry in a hashmap (for instance the email field for a user)
func (rh *HashMap) DelKey(elementid, key string) error {
	conn := rh.pool.Get(rh.dbindex)
	_, err := conn.Do("HDEL", rh.id+":"+elementid, key)
	return err
}

// Remove an element (for instance a user)
func (rh *HashMap) Del(elementid string) error {
	conn := rh.pool.Get(rh.dbindex)
	_, err := conn.Do("DEL", rh.id+":"+elementid)
	return err
}

// Remove this hashmap
func (rh *HashMap) Remove() error {
	conn := rh.pool.Get(rh.dbindex)
	_, err := conn.Do("DEL", rh.id)
	return err
}

/* --- KeyValue functions --- */

// Create a new key/value
func NewKeyValue(pool *ConnectionPool, id string) *KeyValue {
	return &KeyValue{pool, id, 0}
}

// Select a different database
func (rkv *KeyValue) SelectDatabase(dbindex int) {
	rkv.dbindex = dbindex
}

// Set a key and value
func (rkv *KeyValue) Set(key, value string) error {
	conn := rkv.pool.Get(rkv.dbindex)
	_, err := conn.Do("SET", rkv.id+":"+key, value)
	return err
}

// Get a value given a key
func (rkv *KeyValue) Get(key string) (string, error) {
	conn := rkv.pool.Get(rkv.dbindex)
	result, err := redis.String(conn.Do("GET", rkv.id+":"+key))
	if err != nil {
		return "", err
	}
	return result, nil
}

// Remove a key
func (rkv *KeyValue) Del(key string) error {
	conn := rkv.pool.Get(rkv.dbindex)
	_, err := conn.Do("DEL", rkv.id+":"+key)
	return err
}

// Remove this key/value
func (rkv *KeyValue) Remove() error {
	conn := rkv.pool.Get(rkv.dbindex)
	_, err := conn.Do("DEL", rkv.id)
	return err
}

// --- Generic redis functions ---

// Check if a key exists. The key can be a wildcard (ie. "user*").
func hasKey(pool *ConnectionPool, wildcard string, dbindex int) (bool, error) {
	conn := pool.Get(dbindex)
	result, err := redis.Values(conn.Do("KEYS", wildcard))
	if err != nil {
		return false, err
	}
	return len(result) > 0, nil
}
