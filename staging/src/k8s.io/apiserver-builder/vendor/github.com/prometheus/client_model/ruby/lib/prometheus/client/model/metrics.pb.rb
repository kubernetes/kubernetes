## Generated from metrics.proto for io.prometheus.client
require "beefcake"

module Prometheus
  module Client

    module MetricType
      COUNTER = 0
      GAUGE = 1
      SUMMARY = 2
      UNTYPED = 3
      HISTOGRAM = 4
    end

    class LabelPair
      include Beefcake::Message
    end

    class Gauge
      include Beefcake::Message
    end

    class Counter
      include Beefcake::Message
    end

    class Quantile
      include Beefcake::Message
    end

    class Summary
      include Beefcake::Message
    end

    class Untyped
      include Beefcake::Message
    end

    class Histogram
      include Beefcake::Message
    end

    class Bucket
      include Beefcake::Message
    end

    class Metric
      include Beefcake::Message
    end

    class MetricFamily
      include Beefcake::Message
    end

    class LabelPair
      optional :name, :string, 1
      optional :value, :string, 2
    end

    class Gauge
      optional :value, :double, 1
    end

    class Counter
      optional :value, :double, 1
    end

    class Quantile
      optional :quantile, :double, 1
      optional :value, :double, 2
    end

    class Summary
      optional :sample_count, :uint64, 1
      optional :sample_sum, :double, 2
      repeated :quantile, Quantile, 3
    end

    class Untyped
      optional :value, :double, 1
    end

    class Histogram
      optional :sample_count, :uint64, 1
      optional :sample_sum, :double, 2
      repeated :bucket, Bucket, 3
    end

    class Bucket
      optional :cumulative_count, :uint64, 1
      optional :upper_bound, :double, 2
    end

    class Metric
      repeated :label, LabelPair, 1
      optional :gauge, Gauge, 2
      optional :counter, Counter, 3
      optional :summary, Summary, 4
      optional :untyped, Untyped, 5
      optional :histogram, Histogram, 7
      optional :timestamp_ms, :int64, 6
    end

    class MetricFamily
      optional :name, :string, 1
      optional :help, :string, 2
      optional :type, MetricType, 3
      repeated :metric, Metric, 4
    end
  end
end
