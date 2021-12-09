package ratelimiter

import (
	"errors"
	"time"
)

const GC_SIZE int = 100

type Memory struct {
	store           map[string]LeakyBucket
	lastGCCollected time.Time
}

func NewMemory() *Memory {
	m := new(Memory)
	m.store = make(map[string]LeakyBucket)
	m.lastGCCollected = time.Now()
	return m
}

func (m *Memory) GetBucketFor(key string) (*LeakyBucket, error) {

	bucket, ok := m.store[key]
	if !ok {
		return nil, errors.New("miss")
	}

	return &bucket, nil
}

func (m *Memory) SetBucketFor(key string, bucket LeakyBucket) error {

	if len(m.store) > GC_SIZE {
		m.GarbageCollect()
	}

	m.store[key] = bucket

	return nil
}

func (m *Memory) GarbageCollect() {
	now := time.Now()

	// rate limit GC to once per minute
	if now.Add(60*time.Second).Unix() > m.lastGCCollected.Unix() {

		for key, bucket := range m.store {
			// if the bucket is drained, then GC
			if bucket.DrainedAt().Unix() > now.Unix() {
				delete(m.store, key)
			}
		}

		m.lastGCCollected = now
	}
}
