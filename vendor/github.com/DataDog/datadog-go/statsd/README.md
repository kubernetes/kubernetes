## Overview

Package `statsd` provides a Go [dogstatsd](http://docs.datadoghq.com/guides/dogstatsd/) client.  Dogstatsd extends Statsd, adding tags
and histograms.

## Get the code

    $ go get github.com/DataDog/datadog-go/statsd

## Usage

```go
// Create the client
c, err := statsd.New("127.0.0.1:8125")
if err != nil {
    log.Fatal(err)
}
// Prefix every metric with the app name
c.Namespace = "flubber."
// Send the EC2 availability zone as a tag with every metric
c.Tags = append(c.Tags, "us-east-1a")

// Do some metrics!
err = c.Gauge("request.queue_depth", 12, nil, 1)
err = c.Timing("request.duration", duration, nil, 1) // Uses a time.Duration!
err = c.TimeInMilliseconds("request", 12, nil, 1)
err = c.Incr("request.count_total", nil, 1)
err = c.Decr("request.count_total", nil, 1)
err = c.Count("request.count_total", 2, nil, 1)
```

## Buffering Client

DogStatsD accepts packets with multiple statsd payloads in them.  Using the BufferingClient via `NewBufferingClient` will buffer up commands and send them when the buffer is reached or after 100msec.

## Development

Run the tests with:

    $ go test

## Documentation

Please see: http://godoc.org/github.com/DataDog/datadog-go/statsd

## License

go-dogstatsd is released under the [MIT license](http://www.opensource.org/licenses/mit-license.php).

## Credits

Original code by [ooyala](https://github.com/ooyala/go-dogstatsd).
