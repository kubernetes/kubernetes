### Directions
_GitHub Issues are reserved for actionable bug reports and feature requests._
_General questions should be sent to the [InfluxDB mailing list](https://groups.google.com/forum/#!forum/influxdb)._

_Before opening an issue, search for similar bug reports or feature requests on GitHub Issues._
_If no similar issue can be found, fill out either the "Bug Report" or the "Feature Request" section below.
_Erase the other section and everything on and above this line._

### Bug report

__System info:__ [Include InfluxDB version, operating system name, and other relevant details]

__Steps to reproduce:__

1. [First Step]
2. [Second Step]
3. [and so on...]

__Expected behavior:__ [What you expected to happen]

__Actual behavior:__ [What actually happened]

__Additional info:__ [Include gist of relevant config, logs, etc.]

Also, if this is an issue of for performance, locking, etc the following commands are useful to create debug information for the team.

```
curl -o block.txt "http://localhost:8086/debug/pprof/block?debug=1" 
curl -o goroutine.txt "http://localhost:8086/debug/pprof/goroutine?debug=1" 
curl -o heap.txt "http://localhost:8086/debug/pprof/heap?debug=1" 
curl -o vars.txt "http://localhost:8086/debug/vars" 
iostat -xd 1 30 > iostat.txt
influx -execute "show shards" > shards.txt
influx -execute "show stats" > stats.txt
influx -execute "show diagnostics" > diagnostics.txt
```

Please run those if possible and link them from a [gist](http://gist.github.com).

*Please note, the quickest way to fix a bug is to open a Pull Request.*


### Feature Request

Opening a feature request kicks off a discussion.
Requests may be closed if we're not actively planning to work on them.

__Proposal:__ [Description of the feature]

__Current behavior:__ [What currently happens]

__Desired behavior:__ [What you would like to happen]

__Use case:__ [Why is this important (helps with prioritizing requests)]
