install:
	go install ./...

test: install
	go test -v
	toml-test toml-test-decoder
	toml-test -encoder toml-test-encoder

fmt:
	gofmt -w *.go */*.go
	colcheck *.go */*.go

tags:
	find ./ -name '*.go' -print0 | xargs -0 gotags > TAGS

push:
	git push origin master
	git push github master

