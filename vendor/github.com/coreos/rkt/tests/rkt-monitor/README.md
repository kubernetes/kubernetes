# rkt-monitor

This is a small go utility intended to monitor the CPU and memory usage of rkt
and its children processes. This is accomplished by exec'ing rkt, reading proc
once a second for a specified duration, and printing the results.

This utility has a handful of flags:

```
Usage:
  rkt-monitor IMAGE [flags]

Examples:
rkt-monitor mem-stresser.aci -v -d 30s

Flags:
  -f, --to-file[=false]: Save benchmark results to files in a temp dir
  -w, --output-dir="/tmp": Specify directory to write results
  -p, --rkt-dir="": Directory with rkt binary
  -s, --stage1-path="": Path to Stage1 image to use, default: coreos
  -d, --duration="10s": How long to run the ACI
  -h, --help[=false]: help for rkt-monitor
  -r, --repetitions=1: Numbers of benchmark repetitions
  -o, --show-output[=false]: Display rkt's stdout and stderr
  -v, --verbose[=false]: Print current usage every second
```

Some acbuild scripts and golang source code is provided to build ACIs that
attempt to eat up resources in different ways.

An example usage:

```
$ ./tests/rkt-monitor/build-stresser.sh all
Building worker...
Beginning build with an empty ACI
Setting name of ACI to appc.io/rkt-cpu-stresser
Copying host:build-rkt-1.13.0+git/target/bin/cpu-stresser to aci:/worker
Setting exec command [/worker]
Writing ACI to cpu-stresser.aci
Ending the build
Beginning build with an empty ACI
Setting name of ACI to appc.io/rkt-mem-stresser
Copying host:build-rkt-1.13.0+git/target/bin/mem-stresser to aci:/worker
Setting exec command [/worker]
Writing ACI to mem-stresser.aci
Ending the build
Beginning build with an empty ACI
Setting name of ACI to appc.io/rkt-log-stresser
Copying host:build-rkt-1.13.0+git/target/bin/log-stresser to aci:/worker
Setting exec command [/worker]
Writing ACI to log-stresser.aci
Ending the build
$ sudo ./build-rkt-1.13.0+git/target/bin/rkt-monitor log-stresser.aci -r 3 -d 10s
ld-linux-x86-64(29641): seconds alive: 10  avg CPU: 28.948348%  avg Mem: 3 mB  peak Mem: 3 mB
systemd(29698): seconds alive: 10  avg CPU: 0.000000%  avg Mem: 4 mB  peak Mem: 4 mB
systemd-journal(29700): seconds alive: 10  avg CPU: 89.878237%  avg Mem: 7 mB  peak Mem: 7 mB
worker(29705): seconds alive: 10  avg CPU: 8.703743%  avg Mem: 5 mB  peak Mem: 6 mB
load average: Load1: 2.430000 Load5: 1.560000 Load15: 1.100000
container start time: 2539.947085ms
container stop time: 14.724007ms
systemd-journal(29984): seconds alive: 10  avg CPU: 88.553202%  avg Mem: 7 mB  peak Mem: 7 mB
worker(29989): seconds alive: 10  avg CPU: 8.415344%  avg Mem: 5 mB  peak Mem: 6 mB
ld-linux-x86-64(29890): seconds alive: 10  avg CPU: 28.863746%  avg Mem: 3 mB  peak Mem: 3 mB
systemd(29982): seconds alive: 10  avg CPU: 0.000000%  avg Mem: 4 mB  peak Mem: 4 mB
load average: Load1: 2.410000 Load5: 1.600000 Load15: 1.120000
container start time: 2771.857209ms
container stop time: 15.30096ms
systemd(30270): seconds alive: 10  avg CPU: 0.000000%  avg Mem: 4 mB  peak Mem: 4 mB
systemd-journal(30272): seconds alive: 10  avg CPU: 88.863170%  avg Mem: 7 mB  peak Mem: 7 mB
worker(30277): seconds alive: 10  avg CPU: 8.503793%  avg Mem: 5 mB  peak Mem: 6 mB
ld-linux-x86-64(30155): seconds alive: 10  avg CPU: 29.522864%  avg Mem: 3 mB  peak Mem: 3 mB
load average: Load1: 2.270000 Load5: 1.600000 Load15: 1.120000
container start time: 2641.468717ms
container stop time: 14.610641ms
```
