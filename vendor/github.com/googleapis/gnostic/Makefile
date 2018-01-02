
build:	
	go get
	go install
	cd generate-gnostic; go get; go install
	cd apps/report; go get; go install
	cd apps/petstore-builder; go get; go install
	cd plugins/gnostic-go-sample; go get; go install
	cd plugins/gnostic-go-generator/encode-templates; go get; go install
	cd plugins/gnostic-go-generator; go get; go install
	rm -f $(GOPATH)/bin/gnostic-go-client $(GOPATH)/bin/gnostic-go-server
	ln -s $(GOPATH)/bin/gnostic-go-generator $(GOPATH)/bin/gnostic-go-client
	ln -s $(GOPATH)/bin/gnostic-go-generator $(GOPATH)/bin/gnostic-go-server
	cd extensions/sample; make

