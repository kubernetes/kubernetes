cd examples
	ls *.go | xargs -I {} go build {}
	cd ..
go fmt ...swagger && \
go test -test.v ...swagger && \
go install ...swagger && \
go fmt ...restful && \
go test -test.v ...restful && \
go install ...restful