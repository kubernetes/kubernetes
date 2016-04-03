package hh

import (
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/influxdb/influxdb/tsdb"
)

type Processor struct {
	mu sync.RWMutex

	dir            string
	maxSize        int64
	maxAge         time.Duration
	retryRateLimit int64

	queues map[uint64]*queue
	writer shardWriter
	Logger *log.Logger
}

type ProcessorOptions struct {
	MaxSize        int64
	RetryRateLimit int64
}

func NewProcessor(dir string, writer shardWriter, options ProcessorOptions) (*Processor, error) {
	p := &Processor{
		dir:    dir,
		queues: map[uint64]*queue{},
		writer: writer,
		Logger: log.New(os.Stderr, "[handoff] ", log.LstdFlags),
	}
	p.setOptions(options)

	// Create the root directory if it doesn't already exist.
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, fmt.Errorf("mkdir all: %s", err)
	}

	if err := p.loadQueues(); err != nil {
		return p, err
	}
	return p, nil
}

func (p *Processor) setOptions(options ProcessorOptions) {
	p.maxSize = DefaultMaxSize
	if options.MaxSize != 0 {
		p.maxSize = options.MaxSize
	}

	p.retryRateLimit = DefaultRetryRateLimit
	if options.RetryRateLimit != 0 {
		p.retryRateLimit = options.RetryRateLimit
	}
}

func (p *Processor) loadQueues() error {
	files, err := ioutil.ReadDir(p.dir)
	if err != nil {
		return err
	}

	for _, file := range files {
		nodeID, err := strconv.ParseUint(file.Name(), 10, 64)
		if err != nil {
			return err
		}

		if _, err := p.addQueue(nodeID); err != nil {
			return err
		}
	}
	return nil
}

func (p *Processor) addQueue(nodeID uint64) (*queue, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	path := filepath.Join(p.dir, strconv.FormatUint(nodeID, 10))
	if err := os.MkdirAll(path, 0700); err != nil {
		return nil, err
	}

	queue, err := newQueue(path, p.maxSize)
	if err != nil {
		return nil, err
	}
	if err := queue.Open(); err != nil {
		return nil, err
	}
	p.queues[nodeID] = queue
	return queue, nil
}

func (p *Processor) WriteShard(shardID, ownerID uint64, points []tsdb.Point) error {
	queue, ok := p.queues[ownerID]
	if !ok {
		var err error
		if queue, err = p.addQueue(ownerID); err != nil {
			return err
		}
	}

	b := p.marshalWrite(shardID, points)
	return queue.Append(b)
}

func (p *Processor) Process() error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	res := make(chan error, len(p.queues))
	for nodeID, q := range p.queues {
		go func(nodeID uint64, q *queue) {

			// Log how many writes we successfully sent at the end
			var sent int
			start := time.Now()
			defer func(start time.Time) {
				if sent > 0 {
					p.Logger.Printf("%d queued writes sent to node %d in %s", sent, nodeID, time.Since(start))
				}
			}(start)

			limiter := NewRateLimiter(p.retryRateLimit)
			for {
				// Get the current block from the queue
				buf, err := q.Current()
				if err != nil {
					res <- nil
					break
				}

				// unmarshal the byte slice back to shard ID and points
				shardID, points, err := p.unmarshalWrite(buf)
				if err != nil {
					// TODO: If we ever get and error here, we should probably drop the
					// the write and let anti-entropy resolve it.  This would be an urecoverable
					// error and could block the queue indefinitely.
					res <- err
					return
				}

				// Try to send the write to the node
				if err := p.writer.WriteShard(shardID, nodeID, points); err != nil && tsdb.IsRetryable(err) {
					p.Logger.Printf("remote write failed: %v", err)
					res <- nil
					break
				}

				// If we get here, the write succeeded so advance the queue to the next item
				if err := q.Advance(); err != nil {
					res <- err
					return
				}

				sent += 1

				// Update how many bytes we've sent
				limiter.Update(len(buf))

				// Block to maintain the throughput rate
				time.Sleep(limiter.Delay())

			}
		}(nodeID, q)
	}

	for range p.queues {
		err := <-res
		if err != nil {
			return err
		}
	}
	return nil
}

func (p *Processor) marshalWrite(shardID uint64, points []tsdb.Point) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, shardID)
	for _, p := range points {
		b = append(b, []byte(p.String())...)
		b = append(b, '\n')
	}
	return b
}

func (p *Processor) unmarshalWrite(b []byte) (uint64, []tsdb.Point, error) {
	ownerID := binary.BigEndian.Uint64(b[:8])
	points, err := tsdb.ParsePoints(b[8:])
	return ownerID, points, err
}

func (p *Processor) PurgeOlderThan(when time.Duration) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, queue := range p.queues {
		if err := queue.PurgeOlderThan(time.Now().Add(-when)); err != nil {
			return err
		}
	}
	return nil
}
