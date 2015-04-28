.PHONY: all clean editor later nuke todo cover cpu

grep=--include=*.go
target=foo.test

all: editor
	make todo

clean:
	go clean
	rm -f *~

cover:
	t=$(shell tempfile) ; go test -coverprofile $$t && go tool cover -html $$t && unlink $$t

cpu: $(target)
	./$< -noerr -test.cpuprofile cpu.out
	go tool pprof --lines $< cpu.out

editor:
	go fmt
	go test -i
	go test
	go build

later:
	@grep -n $(grep) LATER * || true
	@grep -n $(grep) MAYBE * || true

nuke: clean
	go clean -i

todo:
	@grep -nr $(grep) ^[[:space:]]*_[[:space:]]*=[[:space:]][[:alpha:]][[:alnum:]]* * || true
	@grep -nr $(grep) TODO * || true
	@grep -nr $(grep) BUG * || true
	@grep -nr $(grep) [^[:alpha:]]println * || true
