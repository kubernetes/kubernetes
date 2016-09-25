####Benchmark of the messenger.

```shell
$ go test -v -run=Benckmark* -bench=. 
PASS
BenchmarkMessengerSendSmallMessage	   50000	     70568 ns/op
BenchmarkMessengerSendMediumMessage	   50000	     70265 ns/op
BenchmarkMessengerSendBigMessage	   50000	     72693 ns/op
BenchmarkMessengerSendLargeMessage	   50000	     72896 ns/op
BenchmarkMessengerSendMixedMessage	   50000	     72631 ns/op
BenchmarkMessengerSendRecvSmallMessage	   20000	     78409 ns/op
BenchmarkMessengerSendRecvMediumMessage	   20000	     80471 ns/op
BenchmarkMessengerSendRecvBigMessage	   20000	     82629 ns/op
BenchmarkMessengerSendRecvLargeMessage	   20000	     85987 ns/op
BenchmarkMessengerSendRecvMixedMessage	   20000	     83678 ns/op
ok  	github.com/mesos/mesos-go/messenger	115.135s

$ go test -v -run=Benckmark* -bench=. -cpu=4 -send-routines=4 2>/dev/null
PASS
BenchmarkMessengerSendSmallMessage-4	   50000	     35529 ns/op
BenchmarkMessengerSendMediumMessage-4	   50000	     35997 ns/op
BenchmarkMessengerSendBigMessage-4	   50000	     36871 ns/op
BenchmarkMessengerSendLargeMessage-4	   50000	     37310 ns/op
BenchmarkMessengerSendMixedMessage-4	   50000	     37419 ns/op
BenchmarkMessengerSendRecvSmallMessage-4	   50000	     39320 ns/op
BenchmarkMessengerSendRecvMediumMessage-4	   50000	     41990 ns/op
BenchmarkMessengerSendRecvBigMessage-4	   50000	     42157 ns/op
BenchmarkMessengerSendRecvLargeMessage-4	   50000	     45472 ns/op
BenchmarkMessengerSendRecvMixedMessage-4	   50000	     47393 ns/op
ok  	github.com/mesos/mesos-go/messenger	105.173s
```
 
####environment:

```
OS: Linux yifan-laptop 3.13.0-32-generic #57-Ubuntu SMP Tue Jul 15 03:51:08 UTC 2014 x86_64 x86_64 x86_64 GNU/Linux
CPU: Intel(R) Core(TM) i5-3210M CPU @ 2.50GHz
MEM: 4G DDR3 1600MHz
```
