
EPOCH_TEST_COMMIT	:= 78e6667ae2d67aad100b28ee9580b41b7a24e667
OUTPUT_DIRNAME		?= output
DOC_FILENAME		?= oci-runtime-spec
DOCKER			?= $(shell command -v docker 2>/dev/null)
PANDOC			?= $(shell command -v pandoc 2>/dev/null)
ifeq "$(strip $(PANDOC))" ''
	ifneq "$(strip $(DOCKER))" ''
		PANDOC = $(DOCKER) run \
			-it \
			--rm \
			-v $(shell pwd)/:/input/:ro \
			-v $(shell pwd)/$(OUTPUT_DIRNAME)/:/$(OUTPUT_DIRNAME)/ \
			-u $(shell id -u) \
			vbatts/pandoc
		PANDOC_SRC := /input/
		PANDOC_DST := /
	endif
endif

# These docs are in an order that determines how they show up in the PDF/HTML docs.
DOC_FILES := \
	version.md \
	spec.md \
	principles.md \
	bundle.md \
	runtime.md \
	runtime-linux.md \
	config.md \
	config-linux.md \
	config-solaris.md \
	glossary.md

default: docs

docs: $(OUTPUT_DIRNAME)/$(DOC_FILENAME).pdf $(OUTPUT_DIRNAME)/$(DOC_FILENAME).html

ifeq "$(strip $(PANDOC))" ''
$(OUTPUT_DIRNAME)/$(DOC_FILENAME).pdf $(OUTPUT_DIRNAME)/$(DOC_FILENAME).html:
	$(error cannot build $@ without either pandoc or docker)
else
$(OUTPUT_DIRNAME)/$(DOC_FILENAME).pdf: $(DOC_FILES)
	mkdir -p $(OUTPUT_DIRNAME)/ && \
	$(PANDOC) -f markdown_github -t latex -o $(PANDOC_DST)$@ $(patsubst %,$(PANDOC_SRC)%,$(DOC_FILES))

$(OUTPUT_DIRNAME)/$(DOC_FILENAME).html: $(DOC_FILES)
	mkdir -p $(OUTPUT_DIRNAME)/ && \
	$(PANDOC) -f markdown_github -t html5 -o $(PANDOC_DST)$@ $(patsubst %,$(PANDOC_SRC)%,$(DOC_FILES))
endif

version.md: ./specs-go/version.go
	go run ./.tool/version-doc.go > $@

HOST_GOLANG_VERSION	= $(shell go version | cut -d ' ' -f3 | cut -c 3-)
# this variable is used like a function. First arg is the minimum version, Second arg is the version to be checked.
ALLOWED_GO_VERSION	= $(shell test '$(shell /bin/echo -e "$(1)\n$(2)" | sort -V | head -n1)' = '$(1)' && echo 'true')

test: .govet .golint .gitvalidation

.govet:
	go vet -x ./...

# `go get github.com/golang/lint/golint`
.golint:
ifeq ($(call ALLOWED_GO_VERSION,1.6,$(HOST_GOLANG_VERSION)),true)
	@which golint > /dev/null 2>/dev/null || (echo "ERROR: golint not found. Consider 'make install.tools' target" && false)
	golint ./...
endif


# When this is running in travis, it will only check the travis commit range
.gitvalidation:
	@which git-validation > /dev/null 2>/dev/null || (echo "ERROR: git-validation not found. Consider 'make install.tools' target" && false)
ifdef TRAVIS_COMMIT_RANGE
	git-validation -q -run DCO,short-subject,dangling-whitespace
else
	git-validation -v -run DCO,short-subject,dangling-whitespace -range $(EPOCH_TEST_COMMIT)..HEAD
endif

install.tools: .install.golint .install.gitvalidation

# golint does not even build for <go1.6
.install.golint:
ifeq ($(call ALLOWED_GO_VERSION,1.6,$(HOST_GOLANG_VERSION)),true)
	go get -u github.com/golang/lint/golint
endif

.install.gitvalidation:
	go get -u github.com/vbatts/git-validation

clean:
	rm -rf $(OUTPUT_DIRNAME) *~
	rm -f version.md

