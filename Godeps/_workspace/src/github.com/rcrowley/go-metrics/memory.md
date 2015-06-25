Memory usage
============

(Highly unscientific.)

Command used to gather static memory usage:

```sh
grep ^Vm "/proc/$(ps fax | grep [m]etrics-bench | awk '{print $1}')/status"
```

Program used to gather baseline memory usage:

```go
package main

import "time"

func main() {
	time.Sleep(600e9)
}
```

Baseline
--------

```
VmPeak:    42604 kB
VmSize:    42604 kB
VmLck:         0 kB
VmHWM:      1120 kB
VmRSS:      1120 kB
VmData:    35460 kB
VmStk:       136 kB
VmExe:      1020 kB
VmLib:      1848 kB
VmPTE:        36 kB
VmSwap:        0 kB
```

Program used to gather metric memory usage (with other metrics being similar):

```go
package main

import (
	"fmt"
	"metrics"
	"time"
)

func main() {
	fmt.Sprintf("foo")
	metrics.NewRegistry()
	time.Sleep(600e9)
}
```

1000 counters registered
------------------------

```
VmPeak:    44016 kB
VmSize:    44016 kB
VmLck:         0 kB
VmHWM:      1928 kB
VmRSS:      1928 kB
VmData:    36868 kB
VmStk:       136 kB
VmExe:      1024 kB
VmLib:      1848 kB
VmPTE:        40 kB
VmSwap:        0 kB
```

**1.412 kB virtual, TODO 0.808 kB resident per counter.**

100000 counters registered
--------------------------

```
VmPeak:    55024 kB
VmSize:    55024 kB
VmLck:         0 kB
VmHWM:     12440 kB
VmRSS:     12440 kB
VmData:    47876 kB
VmStk:       136 kB
VmExe:      1024 kB
VmLib:      1848 kB
VmPTE:        64 kB
VmSwap:        0 kB
```

**0.1242 kB virtual, 0.1132 kB resident per counter.**

1000 gauges registered
----------------------

```
VmPeak:    44012 kB
VmSize:    44012 kB
VmLck:         0 kB
VmHWM:      1928 kB
VmRSS:      1928 kB
VmData:    36868 kB
VmStk:       136 kB
VmExe:      1020 kB
VmLib:      1848 kB
VmPTE:        40 kB
VmSwap:        0 kB
```

**1.408 kB virtual, 0.808 kB resident per counter.**

100000 gauges registered
------------------------

```
VmPeak:    55020 kB
VmSize:    55020 kB
VmLck:         0 kB
VmHWM:     12432 kB
VmRSS:     12432 kB
VmData:    47876 kB
VmStk:       136 kB
VmExe:      1020 kB
VmLib:      1848 kB
VmPTE:        60 kB
VmSwap:        0 kB
```

**0.12416 kB virtual, 0.11312 resident per gauge.**

1000 histograms with a uniform sample size of 1028
--------------------------------------------------

```
VmPeak:    72272 kB
VmSize:    72272 kB
VmLck:         0 kB
VmHWM:     16204 kB
VmRSS:     16204 kB
VmData:    65100 kB
VmStk:       136 kB
VmExe:      1048 kB
VmLib:      1848 kB
VmPTE:        80 kB
VmSwap:        0 kB
```

**29.668 kB virtual, TODO 15.084 resident per histogram.**

10000 histograms with a uniform sample size of 1028
---------------------------------------------------

```
VmPeak:   256912 kB
VmSize:   256912 kB
VmLck:         0 kB
VmHWM:    146204 kB
VmRSS:    146204 kB
VmData:   249740 kB
VmStk:       136 kB
VmExe:      1048 kB
VmLib:      1848 kB
VmPTE:       448 kB
VmSwap:        0 kB
```

**21.4308 kB virtual, 14.5084 kB resident per histogram.**

50000 histograms with a uniform sample size of 1028
---------------------------------------------------

```
VmPeak:   908112 kB
VmSize:   908112 kB
VmLck:         0 kB
VmHWM:    645832 kB
VmRSS:    645588 kB
VmData:   900940 kB
VmStk:       136 kB
VmExe:      1048 kB
VmLib:      1848 kB
VmPTE:      1716 kB
VmSwap:     1544 kB
```

**17.31016 kB virtual, 12.88936 kB resident per histogram.**

1000 histograms with an exponentially-decaying sample size of 1028 and alpha of 0.015
-------------------------------------------------------------------------------------

```
VmPeak:    62480 kB
VmSize:    62480 kB
VmLck:         0 kB
VmHWM:     11572 kB
VmRSS:     11572 kB
VmData:    55308 kB
VmStk:       136 kB
VmExe:      1048 kB
VmLib:      1848 kB
VmPTE:        64 kB
VmSwap:        0 kB
```

**19.876 kB virtual, 10.452 kB resident per histogram.**

10000 histograms with an exponentially-decaying sample size of 1028 and alpha of 0.015
--------------------------------------------------------------------------------------

```
VmPeak:   153296 kB
VmSize:   153296 kB
VmLck:         0 kB
VmHWM:    101176 kB
VmRSS:    101176 kB
VmData:   146124 kB
VmStk:       136 kB
VmExe:      1048 kB
VmLib:      1848 kB
VmPTE:       240 kB
VmSwap:        0 kB
```

**11.0692 kB virtual, 10.0056 kB resident per histogram.**

50000 histograms with an exponentially-decaying sample size of 1028 and alpha of 0.015
--------------------------------------------------------------------------------------

```
VmPeak:   557264 kB
VmSize:   557264 kB
VmLck:         0 kB
VmHWM:    501056 kB
VmRSS:    501056 kB
VmData:   550092 kB
VmStk:       136 kB
VmExe:      1048 kB
VmLib:      1848 kB
VmPTE:      1032 kB
VmSwap:        0 kB
```

**10.2932 kB virtual, 9.99872 kB resident per histogram.**

1000 meters
-----------

```
VmPeak:    74504 kB
VmSize:    74504 kB
VmLck:         0 kB
VmHWM:     24124 kB
VmRSS:     24124 kB
VmData:    67340 kB
VmStk:       136 kB
VmExe:      1040 kB
VmLib:      1848 kB
VmPTE:        92 kB
VmSwap:        0 kB
```

**31.9 kB virtual, 23.004 kB resident per meter.**

10000 meters
------------

```
VmPeak:   278920 kB
VmSize:   278920 kB
VmLck:         0 kB
VmHWM:    227300 kB
VmRSS:    227300 kB
VmData:   271756 kB
VmStk:       136 kB
VmExe:      1040 kB
VmLib:      1848 kB
VmPTE:       488 kB
VmSwap:        0 kB
```

**23.6316 kB virtual, 22.618 kB resident per meter.**
