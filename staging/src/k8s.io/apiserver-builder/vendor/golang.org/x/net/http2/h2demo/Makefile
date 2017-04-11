h2demo.linux: h2demo.go
	GOOS=linux go build --tags=h2demo -o h2demo.linux .

FORCE:

upload: FORCE
	go install golang.org/x/build/cmd/upload
	upload --verbose --osarch=linux-amd64 --tags=h2demo --file=go:golang.org/x/net/http2/h2demo --public http2-demo-server-tls/h2demo
