## Introduction

The graphite plugin allows measurements to be saved using the graphite line protocol. By default, enabling the graphite plugin will allow you to collect metrics and store them using the metric name as the measurement.  If you send a metric named `servers.localhost.cpu.loadavg.10`, it will store the full metric name as the measurement with no extracted tags.

While this default setup works, it is not the ideal way to store measurements in InfluxDB since it does not take advantage of tags.  It also will not perform optimally with a large dataset sizes since queries will be forced to use regexes which is known to not scale well.

To extract tags from metrics, one or more templates must be configured to parse metrics into tags and measurements.

## Templates

Templates allow matching parts of a metric name to be used as tag names in the stored metric.  They have a similar format to graphite metric names.  The values in between the separators are used as the tag name.  The location of the tag name that matches the same position as the graphite metric section is used as the value.  If there is no value, the graphite portion is skipped.

The special value _measurement_ is used to define the measurement name.  It can have a trailing `*` to indicate that the remainder of the metric should be used.  If a _measurement_ is not specified, the full metric name is used.

### Basic Matching

`servers.localhost.cpu.loadavg.10`
* Template: `.host.resource.measurement*`
* Output:  _measurement_ =`loading.10` _tags_ =`host=localhost resource=cpu`

### Multiple Measurement Matching

The _measurement_ can be specified multiple times in a template to provide more control over the measurement name.  Multiple values
will be joined together using the _Separator_ config variable.  By default, this value is `.`.

`servers.localhost.cpu.cpu0.user`
* Template: `.host.measurement.cpu.measurement`
* Output: _measurement_ = `cpu.user` _tags_ = `host=localhost cpu=cpu0`

Since '.' requires queries on measurements to be double-quoted, you may want to set this to `_` to simplify querying parsed metrics.

`servers.localhost.cpu.cpu0.user`
* Separator: `_`
* Template: `.host.measurement.cpu.measurement`
* Output: _measurement_ = `cpu_user` _tags_ = `host=localhost cpu=cpu0`

### Adding Tags

Additional tags can be added to a metric that don't exist on the received metric.  You can add additional tags by specifying them after the pattern.  Tags have the same format as the line protocol.  Multiple tags are separated by commas.

`servers.localhost.cpu.loadavg.10`
* Template: `.host.resource.measurement* region=us-west,zone=1a`
* Output:  _measurement_ = `loading.10` _tags_ = `host=localhost resource=cpu region=us-west zone=1a`

## Multiple Templates

One template may not match all metrics.  For example, using multiple plugins with diamond will produce metrics in different formats.  If you need to use multiple templates, you'll need to define a prefix filter that must match before the template can be applied.

### Filters

Filters have a similar format to templates but work more like wildcard expressions.  When multiple filters would match a metric, the more specific one is chosen.  Filters are configured by adding them before the template.

For example,

```
servers.localhost.cpu.loadavg.10
servers.host123.elasticsearch.cache_hits 100
servers.host456.mysql.tx_count 10
```
* `servers.*` would match all values
* `servers.*.mysql` would match `servers.host456.mysql.tx_count 10`
* `servers.localhost.*` would match `servers.localhost.cpu.loadavg`

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

      # default template. Ignore the first graphite component "servers"
     ".measurement*",
 ]
```
