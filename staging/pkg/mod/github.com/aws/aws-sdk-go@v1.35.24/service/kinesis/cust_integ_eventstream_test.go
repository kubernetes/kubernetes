// +build integration

package kinesis_test

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/kinesis"
)

func TestInteg_SubscribeToShard(t *testing.T) {
	desc, err := svc.DescribeStream(&kinesis.DescribeStreamInput{
		StreamName: &streamName,
	})
	if err != nil {
		t.Fatalf("expect no error, %v", err)
	}

	cons, err := svc.DescribeStreamConsumer(
		&kinesis.DescribeStreamConsumerInput{
			StreamARN:    desc.StreamDescription.StreamARN,
			ConsumerName: &consumerName,
		})
	if err != nil {
		t.Fatalf("expect no error, %v", err)
	}

	ctx, cancelFn := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancelFn()

	var recordsRx int32
	var ignoredCount int32
	var goodCount int32
	var wg sync.WaitGroup
	wg.Add(len(desc.StreamDescription.Shards))
	for i, shard := range desc.StreamDescription.Shards {
		go func(idx int, s *kinesis.Shard) {
			defer wg.Done()
			params := &kinesis.SubscribeToShardInput{
				ConsumerARN: cons.ConsumerDescription.ConsumerARN,
				StartingPosition: &kinesis.StartingPosition{
					Type:      aws.String(kinesis.ShardIteratorTypeAtTimestamp),
					Timestamp: &startingTimestamp,
				},
				ShardId: s.ShardId,
			}

			sub, err := svc.SubscribeToShardWithContext(ctx, params)
			if err != nil {
				t.Fatalf("expect no error, %v, %v", err, *s.ShardId)
			}
			defer sub.EventStream.Close()

		Loop:
			for event := range sub.EventStream.Events() {
				switch e := event.(type) {
				case *kinesis.SubscribeToShardEvent:
					if len(e.Records) == 0 {
						atomic.AddInt32(&ignoredCount, 1)
					} else {
						atomic.AddInt32(&goodCount, 1)
						for _, r := range e.Records {
							if len(r.Data) == 0 {
								t.Fatalf("expect data in record, got none")
							}
						}
					}

					newCount := atomic.AddInt32(&recordsRx, int32(len(e.Records)))
					if int(newCount) >= len(records) {
						break Loop
					}
				}
			}

			if err := sub.EventStream.Err(); err != nil {
				t.Fatalf("expect no error, %v, %v", err, *s.ShardId)
			}
		}(i, shard)
	}

	wg.Wait()

	if e, a := len(records), int(recordsRx); e != a {
		t.Errorf("expected to read %v records, got %v", e, a)
	}

	t.Log("Ignored", ignoredCount, "empty events, non-empty", goodCount)
}
