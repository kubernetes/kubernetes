cd examples
	ls *.go | xargs -I {} go build -o /tmp/ignore {}
	cd ..
go fmt ...swagger && \
go test -test.v ...swagger && \
go install ...swagger && \
go fmt ...restful && \
go test -test.v ...restful && \
go install ...restful