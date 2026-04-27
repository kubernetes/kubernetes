# Benchmarking name mangling utilities

```bash
go test -bench XXX -run XXX -benchtime 30s
```

## Benchmarks at b3e7a5386f996177e4808f11acb2aa93a0f660df

```
goos: linux
goarch: amd64
pkg: github.com/go-openapi/swag
cpu: Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz
BenchmarkToXXXName/ToGoName-4         	  862623	     44101 ns/op	   10450 B/op	     732 allocs/op
BenchmarkToXXXName/ToVarName-4        	  853656	     40728 ns/op	   10468 B/op	     734 allocs/op
BenchmarkToXXXName/ToFileName-4       	 1268312	     27813 ns/op	    9785 B/op	     617 allocs/op
BenchmarkToXXXName/ToCommandName-4    	 1276322	     27903 ns/op	    9785 B/op	     617 allocs/op
BenchmarkToXXXName/ToHumanNameLower-4 	  895334	     40354 ns/op	   10472 B/op	     731 allocs/op
BenchmarkToXXXName/ToHumanNameTitle-4 	  882441	     40678 ns/op	   10566 B/op	     749 allocs/op
```

## Benchmarks after PR #79

~ x10 performance improvement and ~ /100 memory allocations.

```
goos: linux
goarch: amd64
pkg: github.com/go-openapi/swag
cpu: Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz
BenchmarkToXXXName/ToGoName-4         	 9595830	      3991 ns/op	      42 B/op	       5 allocs/op
BenchmarkToXXXName/ToVarName-4        	 9194276	      3984 ns/op	      62 B/op	       7 allocs/op
BenchmarkToXXXName/ToFileName-4       	17002711	      2123 ns/op	     147 B/op	       7 allocs/op
BenchmarkToXXXName/ToCommandName-4    	16772926	      2111 ns/op	     147 B/op	       7 allocs/op
BenchmarkToXXXName/ToHumanNameLower-4 	 9788331	      3749 ns/op	      92 B/op	       6 allocs/op
BenchmarkToXXXName/ToHumanNameTitle-4 	 9188260	      3941 ns/op	     104 B/op	       6 allocs/op
```

```
goos: linux
goarch: amd64
pkg: github.com/go-openapi/swag
cpu: AMD Ryzen 7 5800X 8-Core Processor             
BenchmarkToXXXName/ToGoName-16         	18527378	      1972 ns/op	      42 B/op	       5 allocs/op
BenchmarkToXXXName/ToVarName-16        	15552692	      2093 ns/op	      62 B/op	       7 allocs/op
BenchmarkToXXXName/ToFileName-16       	32161176	      1117 ns/op	     147 B/op	       7 allocs/op
BenchmarkToXXXName/ToCommandName-16    	32256634	      1137 ns/op	     147 B/op	       7 allocs/op
BenchmarkToXXXName/ToHumanNameLower-16 	18599661	      1946 ns/op	      92 B/op	       6 allocs/op
BenchmarkToXXXName/ToHumanNameTitle-16 	17581353	      2054 ns/op	     105 B/op	       6 allocs/op
```

## Benchmarks at d7d2d1b895f5b6747afaff312dd2a402e69e818b

go1.24

```
goos: linux
goarch: amd64
pkg: github.com/go-openapi/swag
cpu: AMD Ryzen 7 5800X 8-Core Processor             
BenchmarkToXXXName/ToGoName-16         	19757858	      1881 ns/op	      42 B/op	       5 allocs/op
BenchmarkToXXXName/ToVarName-16        	17494111	      2094 ns/op	      74 B/op	       7 allocs/op
BenchmarkToXXXName/ToFileName-16       	28161226	      1492 ns/op	     158 B/op	       7 allocs/op
BenchmarkToXXXName/ToCommandName-16    	23787333	      1489 ns/op	     158 B/op	       7 allocs/op
BenchmarkToXXXName/ToHumanNameLower-16 	17537257	      2030 ns/op	     103 B/op	       6 allocs/op
BenchmarkToXXXName/ToHumanNameTitle-16 	16977453	      2156 ns/op	     105 B/op	       6 allocs/op
```

## Benchmarks after PR #106

Moving the scope of everything down to a struct allowed to reduce a bit garbage and pooling.

On top of that, ToGoName (and thus ToVarName) have been subject to a minor optimization, removing a few allocations.

Overall timings improve by ~ -10%.

go1.24

```
goos: linux
goarch: amd64
pkg: github.com/go-openapi/swag/mangling
cpu: AMD Ryzen 7 5800X 8-Core Processor             
BenchmarkToXXXName/ToGoName-16         	22496130	      1618 ns/op	      31 B/op	       3 allocs/op
BenchmarkToXXXName/ToVarName-16        	22538068	      1618 ns/op	      33 B/op	       3 allocs/op
BenchmarkToXXXName/ToFileName-16       	27722977	      1236 ns/op	     105 B/op	       6 allocs/op
BenchmarkToXXXName/ToCommandName-16    	27967395	      1258 ns/op	     105 B/op	       6 allocs/op
BenchmarkToXXXName/ToHumanNameLower-16 	18587901	      1917 ns/op	     103 B/op	       6 allocs/op
BenchmarkToXXXName/ToHumanNameTitle-16 	17193208	      2019 ns/op	     108 B/op	       7 allocs/op
```
