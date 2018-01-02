```
derek@proton ~> cat /proc/cpuinfo | grep "model name"
model name  : Intel(R) Core(TM) i7-6500U CPU @ 2.50GHz
model name  : Intel(R) Core(TM) i7-6500U CPU @ 2.50GHz
model name  : Intel(R) Core(TM) i7-6500U CPU @ 2.50GHz
model name  : Intel(R) Core(TM) i7-6500U CPU @ 2.50GHz
derek@proton ~> uname -a
Linux proton 4.4.6 #1-NixOS SMP Wed Mar 16 15:43:17 UTC 2016 x86_64 GNU/Linux
```

## v1.4.0

### log-stresser.aci
```
derek@proton ~/go/src/github.com/coreos/rkt/tests/rkt-monitor> sudo ./rkt-monitor log-stresser.aci
rkt(18493): seconds alive: 10  avg CPU: 28.314541%  avg Mem: 2 mB  peak Mem: 2 mB
systemd(18515): seconds alive: 9  avg CPU: 0.000000%  avg Mem: 4 mB  peak Mem: 4 mB
systemd-journal(18517): seconds alive: 9  avg CPU: 88.397098%  avg Mem: 7 mB  peak Mem: 7 mB
worker(18521): seconds alive: 9  avg CPU: 7.330367%  avg Mem: 5 mB  peak Mem: 6 mB
load average: Load1: 0.390000 Load5: 0.120000 Load15: 0.080000
container start time: 250721ns
container stop time: 17332926ns

```

### mem-stresser.aci
```
derek@proton ~/go/src/github.com/coreos/rkt/tests/rkt-monitor> sudo ./rkt-monitor mem-stresser.aci
worker(18634): seconds alive: 9  avg CPU: 98.550401%  avg Mem: 318 mB  peak Mem: 555 mB
rkt(18599): seconds alive: 10  avg CPU: 3.583814%  avg Mem: 2 mB  peak Mem: 2 mB
systemd(18628): seconds alive: 9  avg CPU: 0.000000%  avg Mem: 4 mB  peak Mem: 4 mB
systemd-journal(18630): seconds alive: 9  avg CPU: 0.000000%  avg Mem: 6 mB  peak Mem: 6 mB
load average: Load1: 0.310000 Load5: 0.150000 Load15: 0.090000
container start time: 259746ns
container stop time: 17593446ns
```

### cpu-stresser.aci
```
derek@proton ~/go/src/github.com/coreos/rkt/tests/rkt-monitor> sudo ./rkt-monitor cpu-stresser.aci
rkt(18706): seconds alive: 10  avg CPU: 3.587050%  avg Mem: 2 mB  peak Mem: 2 mB
systemd(18736): seconds alive: 9  avg CPU: 0.000000%  avg Mem: 4 mB  peak Mem: 4 mB
systemd-journal(18740): seconds alive: 9  avg CPU: 0.000000%  avg Mem: 6 mB  peak Mem: 6 mB
worker(18744): seconds alive: 9  avg CPU: 88.937493%  avg Mem: 808 kB  peak Mem: 808 kB
load average: Load1: 0.310000 Load5: 0.130000 Load15: 0.080000
container start time: 296570ns
container stop time: 16124700ns
```

### too-many-apps.podmanifest
```
derek@proton ~/go/src/github.com/coreos/rkt/tests/rkt-monitor> sudo ./rkt-monitor too-many-apps.podmanifest -d 30s
# Identical (aside from PID) worker-binary lines removed
rkt(17227): seconds alive: 20  avg CPU: 9.595387%  avg Mem: 3 mB  peak Mem: 20 mB
systemd(17253): seconds alive: 17  avg CPU: 0.329028%  avg Mem: 16 mB  peak Mem: 16 mB
systemd-journal(17255): seconds alive: 17  avg CPU: 0.000000%  avg Mem: 6 mB  peak Mem: 6 mB
worker-binary(17883): seconds alive: 17  avg CPU: 0.000000%  avg Mem: 840 kB  peak Mem: 840 kB
load average: Load1: 0.480000 Load5: 0.350000 Load15: 0.300000
container start time: 528476ns
container stop time: 4522346ns
```

