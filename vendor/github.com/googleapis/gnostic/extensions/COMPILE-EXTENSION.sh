go get github.com/golang/protobuf/protoc-gen-go

protoc \
--go_out=Mgoogle/protobuf/any.proto=github.com/golang/protobuf/ptypes/any:. *.proto 

