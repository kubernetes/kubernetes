# Example

Demonstrates how the Go standard library `httptrace` can be used with the SDK
to collect HTTP request tracing timing using the SDK's API operation methods
like SNS's `PublishWithContext`.

The `trace.go` file demonstrates how the `httptrace` package's `ClientTrace`
can be created to gather timing information from HTTP requests made.

The `config.go` file provides additional configuration settings to control how
the HTTP client and its transport is configured. Such as, timeouts, and
keepalive.

## Usage

Run the example providing your SNS topic's ARN as the `-topic` parameter. This
example assumes that the region is provided via the environment variable and
the AWS shared credentials file (~/.aws/credentials)'s `default` provide
provides credentials.

```sh
AWS_REGION=us-west-2 go run -tags example . -topic arn:aws:sns:us-west-2:0123456789:mytopicname
```

Once the example starts you'll be prompted with a `Message:` statement. Input
the message that you'd like to send to the topic on a single line and hit
`enter` to send it.

```
Message: My Really cool Message
```

The example will output the http trace timing information for how long the request took.

```
2020/07/21 15:39:07 Latency: 278.508656ms, Validate: 19.515µs, Build: 190.755µs, Attempts: 1,
	Attempt: 0, Latency: 278.240054ms, Sign: 453.163µs, Send: 277.580235ms, Unmarshal: 202.311µs, WillRetry: false,
		HTTP: Latency: 277.580856ms, ConnReused: false, GetConn: 225.662398ms, WriteRequest: 325.956µs, WaitResponseFirstByte: 277.509316ms, ReadResponseHeader: 71.234µs,
			Conn: DNS: 20.385136ms, Connect: 14.947772ms, TLS: 189.910822ms,

Message: second
2020/07/21 15:39:09 Latency: 101.936094ms, Validate: 2.644µs, Build: 67.157µs, Attempts: 1,
	Attempt: 0, Latency: 101.836442ms, Sign: 122.098µs, Send: 101.517516ms, Unmarshal: 191.31µs, WillRetry: false,
		HTTP: Latency: 101.518147ms, ConnReused: true, GetConn: 38.265µs, WriteRequest: 178.058µs, WaitResponseFirstByte: 101.457147ms, ReadResponseHeader: 60.526µs,

Message: third
2020/07/21 15:39:10 Latency: 32.373919ms, Validate: 2.998µs, Build: 39.136µs, Attempts: 1,
	Attempt: 0, Latency: 32.295677ms, Sign: 104.978µs, Send: 32.040306ms, Unmarshal: 146.096µs, WillRetry: false,
		HTTP: Latency: 32.04078ms, ConnReused: true, GetConn: 33.36µs, WriteRequest: 166.508µs, WaitResponseFirstByte: 31.980933ms, ReadResponseHeader: 59.235µs,

Message: fourth
2020/07/21 15:39:13 Latency: 33.353819ms, Validate: 2.002µs, Build: 36.807µs, Attempts: 1,
	Attempt: 0, Latency: 33.29846ms, Sign: 70.238µs, Send: 33.125914ms, Unmarshal: 98.181µs, WillRetry: false,
		HTTP: Latency: 33.126453ms, ConnReused: true, GetConn: 47.516µs, WriteRequest: 251.875µs, WaitResponseFirstByte: 33.05992ms, ReadResponseHeader: 66.017µs,
```

