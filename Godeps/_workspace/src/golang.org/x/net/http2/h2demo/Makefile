h2demo.linux: h2demo.go
	GOOS=linux go build --tags=h2demo -o h2demo.linux .

upload: h2demo.linux
	cat h2demo.linux | go run launch.go --write_object=http2-demo-server-tls/h2demo --write_object_is_public
