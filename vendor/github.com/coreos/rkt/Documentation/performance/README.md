# Benchmarks

rkt has a utility called rkt-monitor that will run rkt with an example
workload, and track the memory and CPU usage. It does this by exec'ing rkt with
an ACI or pod manifest, watching the resource consumption for rkt and all
children processes, and after a timeout killing rkt and printing the results.

## Running the Benchmarks

To run the benchmarks, one must have both a built version of rkt-monitor and an
ACI or pod manifest. Additionally, rkt must be available on the `PATH`.

To build rkt-monitor, `cd` to `tests/rkt-monitor` and run the `build` script in
that directory.

To build one of the provided workloads, run any of the `build-*` scripts in
`tests/rkt-monitor`. All scripts require acbuild to be available on the current
`PATH`. The script will produce either an ACI, or a directory with multiple
ACIs and a pod manifest. In the case of the latter, the ACIs in the created
directory must be imported into rkt's cas before running rkt-monitor, via the
command `rkt fetch --insecure-options=image <newDirectory>/*`.

With rkt-monitor and an ACI or a pod manifest, now the benchmarks can be run
via `./rkt-monitor <workload>`.

There are four flags available to influence how rkt-monitor runs. `-r ` set the
number of benchmark experiment repetitions, `-f` save output to files in a
temporary directory with `rkt_benchmark` prefix, `-v` will print out the current
resource usage of each process every second. `-d` can be used to specify a
duration to run the tests for (default of 10s). For example, `-d 30s` will run
the tests for 30 seconds.

# Profiling

rkt will provide two hidden global flags `--cpuprofile` and `--memprofile` that can be used for performance profiling.
Setting `--cpuprofile=$FILE` will make rkt write down the CPU profiles to the `$FILE`.
Setting `--memprofile=$FILE` will make rkt write down the Memory profile to the `$FILE`.
Note that memory profile will only be written down before rkt exits, so during the execution
of rkt, the memory profile will be empty.

The profile result can be viewed by go's profiling tool, for example:

```shell
$ sudo /usr/bin/rkt --cpuprofile=/tmp/cpu.profile --memprofile=/tmp/mem.profile gc --grace-period=0
$ go tool pprof /usr/bin/rkt /tmp/cpu.profile
$ go tool pprof /usr/bin/rkt /tmp/mem.profile
```

For more profiling tips, please see [Profiling Go Programs][profiling].


[profiling]: https://blog.golang.org/profiling-go-programs
