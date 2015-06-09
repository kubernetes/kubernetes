del ff_ffjson.go
ffjson ff.go

go test -benchmem -bench MarshalJSON

REM ### Bench CPU ###
rem go test -benchmem -test.run=none -bench MarshalJSONNative -cpuprofile="cpu.dat" -benchtime 10s &&go tool pprof -gif tests.test.exe cpu.dat >out.gif


