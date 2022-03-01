package ratelimiter

type Storage interface {
	GetBucketFor(string) (*LeakyBucket, error)
	SetBucketFor(string, LeakyBucket) error
}
