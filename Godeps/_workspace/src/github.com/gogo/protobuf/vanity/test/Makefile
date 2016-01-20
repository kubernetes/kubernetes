regenerate:
	go install github.com/gogo/protobuf/protoc-gen-gogofast
	protoc --gogofast_out=./fast/ vanity.proto
	protoc --proto_path=../../:../../../../../:../../protobuf/:. --gogofast_out=./fast/ gogovanity.proto
	protoc-min-version -version="3.0.0" --proto_path=../../:../../../../../:../../protobuf/:. --gogofast_out=./fast/ proto3.proto
	go install github.com/gogo/protobuf/protoc-gen-gogofaster
	protoc --gogofaster_out=./faster/ vanity.proto
	protoc --proto_path=../../:../../../../../:../../protobuf/:. --gogofaster_out=./faster/ gogovanity.proto
	protoc-min-version -version="3.0.0" --proto_path=../../:../../../../../:../../protobuf/:. --gogofaster_out=./faster/ proto3.proto
	go install github.com/gogo/protobuf/protoc-gen-gogoslick
	protoc --gogoslick_out=./slick/ vanity.proto
	protoc --proto_path=../../:../../../../../:../../protobuf/:. --gogoslick_out=./slick/ gogovanity.proto
	protoc-min-version -version="3.0.0" --proto_path=../../:../../../../../:../../protobuf/:. --gogoslick_out=./slick/ proto3.proto

test:
	go install github.com/gogo/protobuf/protoc-gen-gofast
	protoc --gofast_out=./gofast/ vanity.proto
	go test ./...
