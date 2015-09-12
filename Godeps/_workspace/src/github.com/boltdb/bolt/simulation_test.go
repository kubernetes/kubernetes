package bolt_test

import (
	"bytes"
	"fmt"
	"math/rand"
	"sync"
	"testing"

	"github.com/boltdb/bolt"
)

func TestSimulate_1op_1p(t *testing.T)     { testSimulate(t, 100, 1) }
func TestSimulate_10op_1p(t *testing.T)    { testSimulate(t, 10, 1) }
func TestSimulate_100op_1p(t *testing.T)   { testSimulate(t, 100, 1) }
func TestSimulate_1000op_1p(t *testing.T)  { testSimulate(t, 1000, 1) }
func TestSimulate_10000op_1p(t *testing.T) { testSimulate(t, 10000, 1) }

func TestSimulate_10op_10p(t *testing.T)    { testSimulate(t, 10, 10) }
func TestSimulate_100op_10p(t *testing.T)   { testSimulate(t, 100, 10) }
func TestSimulate_1000op_10p(t *testing.T)  { testSimulate(t, 1000, 10) }
func TestSimulate_10000op_10p(t *testing.T) { testSimulate(t, 10000, 10) }

func TestSimulate_100op_100p(t *testing.T)   { testSimulate(t, 100, 100) }
func TestSimulate_1000op_100p(t *testing.T)  { testSimulate(t, 1000, 100) }
func TestSimulate_10000op_100p(t *testing.T) { testSimulate(t, 10000, 100) }

func TestSimulate_10000op_1000p(t *testing.T) { testSimulate(t, 10000, 1000) }

// Randomly generate operations on a given database with multiple clients to ensure consistency and thread safety.
func testSimulate(t *testing.T, threadCount, parallelism int) {
	if testing.Short() {
		t.Skip("skipping test in short mode.")
	}

	rand.Seed(int64(qseed))

	// A list of operations that readers and writers can perform.
	var readerHandlers = []simulateHandler{simulateGetHandler}
	var writerHandlers = []simulateHandler{simulateGetHandler, simulatePutHandler}

	var versions = make(map[int]*QuickDB)
	versions[1] = NewQuickDB()

	db := NewTestDB()
	defer db.Close()

	var mutex sync.Mutex

	// Run n threads in parallel, each with their own operation.
	var wg sync.WaitGroup
	var threads = make(chan bool, parallelism)
	var i int
	for {
		threads <- true
		wg.Add(1)
		writable := ((rand.Int() % 100) < 20) // 20% writers

		// Choose an operation to execute.
		var handler simulateHandler
		if writable {
			handler = writerHandlers[rand.Intn(len(writerHandlers))]
		} else {
			handler = readerHandlers[rand.Intn(len(readerHandlers))]
		}

		// Execute a thread for the given operation.
		go func(writable bool, handler simulateHandler) {
			defer wg.Done()

			// Start transaction.
			tx, err := db.Begin(writable)
			if err != nil {
				t.Fatal("tx begin: ", err)
			}

			// Obtain current state of the dataset.
			mutex.Lock()
			var qdb = versions[tx.ID()]
			if writable {
				qdb = versions[tx.ID()-1].Copy()
			}
			mutex.Unlock()

			// Make sure we commit/rollback the tx at the end and update the state.
			if writable {
				defer func() {
					mutex.Lock()
					versions[tx.ID()] = qdb
					mutex.Unlock()

					ok(t, tx.Commit())
				}()
			} else {
				defer tx.Rollback()
			}

			// Ignore operation if we don't have data yet.
			if qdb == nil {
				return
			}

			// Execute handler.
			handler(tx, qdb)

			// Release a thread back to the scheduling loop.
			<-threads
		}(writable, handler)

		i++
		if i > threadCount {
			break
		}
	}

	// Wait until all threads are done.
	wg.Wait()
}

type simulateHandler func(tx *bolt.Tx, qdb *QuickDB)

// Retrieves a key from the database and verifies that it is what is expected.
func simulateGetHandler(tx *bolt.Tx, qdb *QuickDB) {
	// Randomly retrieve an existing exist.
	keys := qdb.Rand()
	if len(keys) == 0 {
		return
	}

	// Retrieve root bucket.
	b := tx.Bucket(keys[0])
	if b == nil {
		panic(fmt.Sprintf("bucket[0] expected: %08x\n", trunc(keys[0], 4)))
	}

	// Drill into nested buckets.
	for _, key := range keys[1 : len(keys)-1] {
		b = b.Bucket(key)
		if b == nil {
			panic(fmt.Sprintf("bucket[n] expected: %v -> %v\n", keys, key))
		}
	}

	// Verify key/value on the final bucket.
	expected := qdb.Get(keys)
	actual := b.Get(keys[len(keys)-1])
	if !bytes.Equal(actual, expected) {
		fmt.Println("=== EXPECTED ===")
		fmt.Println(expected)
		fmt.Println("=== ACTUAL ===")
		fmt.Println(actual)
		fmt.Println("=== END ===")
		panic("value mismatch")
	}
}

// Inserts a key into the database.
func simulatePutHandler(tx *bolt.Tx, qdb *QuickDB) {
	var err error
	keys, value := randKeys(), randValue()

	// Retrieve root bucket.
	b := tx.Bucket(keys[0])
	if b == nil {
		b, err = tx.CreateBucket(keys[0])
		if err != nil {
			panic("create bucket: " + err.Error())
		}
	}

	// Create nested buckets, if necessary.
	for _, key := range keys[1 : len(keys)-1] {
		child := b.Bucket(key)
		if child != nil {
			b = child
		} else {
			b, err = b.CreateBucket(key)
			if err != nil {
				panic("create bucket: " + err.Error())
			}
		}
	}

	// Insert into database.
	if err := b.Put(keys[len(keys)-1], value); err != nil {
		panic("put: " + err.Error())
	}

	// Insert into in-memory database.
	qdb.Put(keys, value)
}

// QuickDB is an in-memory database that replicates the functionality of the
// Bolt DB type except that it is entirely in-memory. It is meant for testing
// that the Bolt database is consistent.
type QuickDB struct {
	sync.RWMutex
	m map[string]interface{}
}

// NewQuickDB returns an instance of QuickDB.
func NewQuickDB() *QuickDB {
	return &QuickDB{m: make(map[string]interface{})}
}

// Get retrieves the value at a key path.
func (db *QuickDB) Get(keys [][]byte) []byte {
	db.RLock()
	defer db.RUnlock()

	m := db.m
	for _, key := range keys[:len(keys)-1] {
		value := m[string(key)]
		if value == nil {
			return nil
		}
		switch value := value.(type) {
		case map[string]interface{}:
			m = value
		case []byte:
			return nil
		}
	}

	// Only return if it's a simple value.
	if value, ok := m[string(keys[len(keys)-1])].([]byte); ok {
		return value
	}
	return nil
}

// Put inserts a value into a key path.
func (db *QuickDB) Put(keys [][]byte, value []byte) {
	db.Lock()
	defer db.Unlock()

	// Build buckets all the way down the key path.
	m := db.m
	for _, key := range keys[:len(keys)-1] {
		if _, ok := m[string(key)].([]byte); ok {
			return // Keypath intersects with a simple value. Do nothing.
		}

		if m[string(key)] == nil {
			m[string(key)] = make(map[string]interface{})
		}
		m = m[string(key)].(map[string]interface{})
	}

	// Insert value into the last key.
	m[string(keys[len(keys)-1])] = value
}

// Rand returns a random key path that points to a simple value.
func (db *QuickDB) Rand() [][]byte {
	db.RLock()
	defer db.RUnlock()
	if len(db.m) == 0 {
		return nil
	}
	var keys [][]byte
	db.rand(db.m, &keys)
	return keys
}

func (db *QuickDB) rand(m map[string]interface{}, keys *[][]byte) {
	i, index := 0, rand.Intn(len(m))
	for k, v := range m {
		if i == index {
			*keys = append(*keys, []byte(k))
			if v, ok := v.(map[string]interface{}); ok {
				db.rand(v, keys)
			}
			return
		}
		i++
	}
	panic("quickdb rand: out-of-range")
}

// Copy copies the entire database.
func (db *QuickDB) Copy() *QuickDB {
	db.RLock()
	defer db.RUnlock()
	return &QuickDB{m: db.copy(db.m)}
}

func (db *QuickDB) copy(m map[string]interface{}) map[string]interface{} {
	clone := make(map[string]interface{}, len(m))
	for k, v := range m {
		switch v := v.(type) {
		case map[string]interface{}:
			clone[k] = db.copy(v)
		default:
			clone[k] = v
		}
	}
	return clone
}

func randKey() []byte {
	var min, max = 1, 1024
	n := rand.Intn(max-min) + min
	b := make([]byte, n)
	for i := 0; i < n; i++ {
		b[i] = byte(rand.Intn(255))
	}
	return b
}

func randKeys() [][]byte {
	var keys [][]byte
	var count = rand.Intn(2) + 2
	for i := 0; i < count; i++ {
		keys = append(keys, randKey())
	}
	return keys
}

func randValue() []byte {
	n := rand.Intn(8192)
	b := make([]byte, n)
	for i := 0; i < n; i++ {
		b[i] = byte(rand.Intn(255))
	}
	return b
}
