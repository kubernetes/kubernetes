go test -test.v ...restful && \
go test -test.v ...swagger && \
go vet ...restful && \
go fmt ...swagger && \
go install ...swagger && \
go fmt ...restful && \
go install ...restful
cd examples
	ls *.go | xargs -I {} go build -o /tmp/ignore {}
	cd ..