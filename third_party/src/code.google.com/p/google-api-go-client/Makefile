all:
	go install code.google.com/p/google-api-go-client/googleapi
	go install code.google.com/p/google-api-go-client/google-api-go-generator
	$(GOPATH)/bin/google-api-go-generator -cache=false -install -api=*

cached:
	go install code.google.com/p/google-api-go-client/googleapi
	go install code.google.com/p/google-api-go-client/google-api-go-generator
	$(GOPATH)/bin/google-api-go-generator -cache=true -install -api=*
