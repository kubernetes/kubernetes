
build:
	generate-gnostic --extension x-sampleone.json --out_dir=generated
	cd generated/gnostic-x-sampleone/proto; protoc --go_out=. *.proto
	cd generated/gnostic-x-sampleone; go get; go install
	generate-gnostic --extension x-sampletwo.json --out_dir=generated
	cd generated/gnostic-x-sampletwo/proto; protoc --go_out=. *.proto
	cd generated/gnostic-x-sampletwo; go get; go install
