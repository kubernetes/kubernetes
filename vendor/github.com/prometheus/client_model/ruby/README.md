# Prometheus Ruby client model

Data model artifacts for the [Prometheus Ruby client][1].

## Installation

    gem install prometheus-client-model

## Usage

Build the artifacts from the protobuf specification:

    make build

While this Gem's main purpose is to define the Prometheus data types for the
[client][1], it's possible to use it without the client to decode a stream of
delimited protobuf messages:

```ruby
require 'open-uri'
require 'prometheus/client/model'

CONTENT_TYPE = 'application/vnd.google.protobuf; proto=io.prometheus.client.MetricFamily; encoding=delimited'

stream = open('http://localhost:9090/metrics', 'Accept' => CONTENT_TYPE).read
while family = Prometheus::Client::MetricFamily.read_delimited(stream)
  puts family
end
```

[1]: https://github.com/prometheus/client_ruby
