package runtime

import "io"

// DiscardConsumer does absolutely nothing, it's a black hole.
var DiscardConsumer = ConsumerFunc(func(_ io.Reader, _ interface{}) error { return nil })

// DiscardProducer does absolutely nothing, it's a black hole.
var DiscardProducer = ProducerFunc(func(_ io.Writer, _ interface{}) error { return nil })
