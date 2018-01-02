# The Graphite Input

## A Note On UDP/IP OS Buffer Sizes

If you're using UDP input and running Linux or FreeBSD, please adjust your UDP buffer
size limit, [see here for more details.](../udp/README.md#a-note-on-udpip-os-buffer-sizes)

## Configuration

Each Graphite input allows the binding address, target database, and protocol to be set. If the database does not exist, it will be created automatically when the input is initialized. The write-consistency-level can also be set. If any write operations do not meet the configured consistency guarantees, an error will occur and the data will not be indexed. The default consistency-level is `ONE`.

Each Graphite input also performs internal batching of the points it receives, as batched writes to the database are more efficient. The default _batch size_ is 1000, _pending batch_ factor is 5, with a _batch timeout_ of 1 second. This means the input will write batches of maximum size 1000, but if a batch has not reached 1000 points within 1 second of the first point being added to a batch, it will emit that batch regardless of size. The pending batch factor controls how many batches can be in memory at once, allowing the input to transmit a batch, while still building other batches.

## Parsing Metrics

The Graphite plugin allows measurements to be saved using the Graphite line protocol. By default, enabling the Graphite plugin will allow you to collect metrics and store them using the metric name as the measurement.  If you send a metric named `servers.localhost.cpu.loadavg.10`, it will store the full metric name as the measurement with no extracted tags.

While this default setup works, it is not the ideal way to store measurements in InfluxDB since it does not take advantage of tags.  It also will not perform optimally with large dataset sizes since queries will be forced to use regexes which is known to not scale well.

To extract tags from metrics, one or more templates must be configured to parse metrics into tags and measurements.

## Templates

Templates allow matching parts of a metric name to be used as tag keys in the stored metric.  They have a similar format to Graphite metric names.  The values in between the separators are used as the tag keys.  The location of the tag key that matches the same position as the Graphite metric section is used as the value.  If there is no value, the Graphite portion is skipped.

The special value _measurement_ is used to define the measurement name.  It can have a trailing `*` to indicate that the remainder of the metric should be used.  If a _measurement_ is not specified, the full metric name is used.

### Basic Matching

`servers.localhost.cpu.loadavg.10`
* Template: `.host.resource.measurement*`
* Output:  _measurement_ =`loadavg.10` _tags_ =`host=localhost resource=cpu`

### Multiple Measurement & Tags Matching

The _measurement_ can be specified multiple times in a template to provide more control over the measurement name. Tags can also be
matched multiple times. Multiple values will be joined together using the _Separator_ config variable.  By default, this value is `.`.

`servers.localhost.localdomain.cpu.cpu0.user`
* Template: `.host.host.measurement.cpu.measurement`
* Output: _measurement_ = `cpu.user` _tags_ = `host=localhost.localdomain cpu=cpu0`

Since `.` requires queries on measurements to be double-quoted, you may want to set this to `_` to simplify querying parsed metrics.

`servers.localhost.cpu.cpu0.user`
* Separator: `_`
* Template: `.host.measurement.cpu.measurement`
* Output: _measurement_ = `cpu_user` _tags_ = `host=localhost cpu=cpu0`

### Adding Tags

Additional tags can be added to a metric if they don't exist on the received metric.  You can add additional tags by specifying them after the pattern.  Tags have the same format as the line protocol.  Multiple tags are separated by commas.

`servers.localhost.cpu.loadavg.10`
* Template: `.host.resource.measurement* region=us-west,zone=1a`
* Output:  _measurement_ = `loadavg.10` _tags_ = `host=localhost resource=cpu region=us-west zone=1a`

### Fields

A field key can be specified by using the keyword _field_. By default if no _field_ keyword is specified then the metric will be written to a field named _value_.

The field key can also be derived from the second "half" of the input metric-name by specifying ```field*``` (eg ```measurement.measurement.field*```). This cannot be used in conjunction with "measurement*"!

It's possible to amend measurement metrics with additional fields, e.g:

Input:
```
sensu.metric.net.server0.eth0.rx_packets 461295119435 1444234982
sensu.metric.net.server0.eth0.tx_bytes 1093086493388480 1444234982
sensu.metric.net.server0.eth0.rx_bytes 1015633926034834 1444234982
sensu.metric.net.server0.eth0.tx_errors 0 1444234982
sensu.metric.net.server0.eth0.rx_errors 0 1444234982
sensu.metric.net.server0.eth0.tx_dropped 0 1444234982
sensu.metric.net.server0.eth0.rx_dropped 0 1444234982
```

With template:
```
sensu.metric.* ..measurement.host.interface.field
```

Becomes database entry:
```
> select * from net
name: net
---------
time      host  interface rx_bytes    rx_dropped  rx_errors rx_packets    tx_bytes    tx_dropped  tx_errors
1444234982000000000 server0  eth0    1.015633926034834e+15 0   0   4.61295119435e+11 1.09308649338848e+15  0 0
```

## Multiple Templates

One template may not match all metrics.  For example, using multiple plugins with diamond will produce metrics in different formats.  If you need to use multiple templates, you'll need to define a prefix filter that must match before the template can be applied.

### Filters

Filters have a similar format to templates but work more like wildcard expressions.  When multiple filters would match a metric, the more specific one is chosen.  Filters are configured by adding them before the template.

For example,

```
servers.localhost.cpu.loadavg.10
servers.host123.elasticsearch.cache_hits 100
servers.host456.mysql.tx_count 10
servers.host789.prod.mysql.tx_count 10
```
* `servers.*` would match all values
* `servers.*.mysql` would match `servers.host456.mysql.tx_count 10`
* `servers.localhost.*` would match `servers.localhost.cpu.loadavg`
* `servers.*.*.mysql` would match `servers.host789.prod.mysql.tx_count 10`

## Default Templates

If no template filters are defined or you want to just have one basic template, you can define a default template.  This template will apply to any metric that has not already matched a filter.

```
dev.http.requests.200
prod.myapp.errors.count
dev.db.queries.count
```

* `env.app.measurement*` would create
  * _measurement_=`requests.200` _tags_=`env=dev,app=http`
  * _measurement_= `errors.count` _tags_=`env=prod,app=myapp`
  * _measurement_=`queries.count` _tags_=`env=dev,app=db`

## Global Tags

If you need to add the same set of tags to all metrics, you can define them globally at the plugin level and not within each template description.

## Minimal Config
```
[[graphite]]
  enabled = true
  # bind-address = ":2003"
  # protocol = "tcp"
  # consistency-level = "one"

  ### If matching multiple measurement files, this string will be used to join the matched values.
  # separator = "."

  ### Default tags that will be added to all metrics.  These can be overridden at the template level
  ### or by tags extracted from metric
  # tags = ["region=us-east", "zone=1c"]

  ### Each template line requires a template pattern.  It can have an optional
  ### filter before the template and separated by spaces.  It can also have optional extra
  ### tags following the template.  Multiple tags should be separated by commas and no spaces
  ### similar to the line protocol format.  The can be only one default template.
  # templates = [
  #   "*.app env.service.resource.measurement",
  #   # Default template
  #   "server.*",
 #]
```

## Customized Config
```
[[graphite]]
   enabled = true
   separator = "_"
   tags = ["region=us-east", "zone=1c"]
   templates = [
     # filter + template
     "*.app env.service.resource.measurement",

     # filter + template + extra tag
     "stats.* .host.measurement* region=us-west,agent=sensu",

     # filter + template with field key
     "stats.* .host.measurement.field",

     # default template. Ignore the first Graphite component "servers"
     ".measurement*",
 ]
```

## Two Graphite Listeners, UDP & TCP, Config

```
[[graphite]]
  enabled = true
  bind-address = ":2003"
  protocol = "tcp"
  # consistency-level = "one"

[[graphite]]
  enabled = true
  bind-address = ":2004" # the bind address
  protocol = "udp" # protocol to read via
  udp-read-buffer = 8388608 # (8*1024*1024) UDP read buffer size
```
