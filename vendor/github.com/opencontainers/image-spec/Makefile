GO15VENDOREXPERIMENT=1
export GO15VENDOREXPERIMENT

DOCKER ?= $(shell command -v docker 2>/dev/null)
PANDOC ?= $(shell command -v pandoc 2>/dev/null)

ifeq "$(strip $(PANDOC))" ''
	ifneq "$(strip $(DOCKER))" ''
		PANDOC = $(DOCKER) run \
			-it \
			--rm \
			-v $(shell pwd)/:/input/:ro \
			-v $(shell pwd)/$(OUTPUT_DIRNAME)/:/$(OUTPUT_DIRNAME)/ \
			-u $(shell id -u) \
			--workdir /input \
			docker.io/vbatts/pandoc:1.17.0.3-2.fc25.x86_64
		PANDOC_SRC := /input/
		PANDOC_DST := /
	endif
endif

# These docs are in an order that determines how they show up in the PDF/HTML docs.
DOC_FILES := \
	spec.md \
	media-types.md \
	descriptor.md \
	image-layout.md \
	manifest.md \
	image-index.md \
	layer.md \
	config.md \
	annotations.md \
	conversion.md \
	considerations.md \
	implementations.md

FIGURE_FILES := \
	img/media-types.png

OUTPUT_DIRNAME		?= output/
DOC_FILENAME	?= oci-image-spec

EPOCH_TEST_COMMIT ?= v0.2.0

TOOLS := esc gitvalidation glide glide-vc

default: check-license lint test

help:
	@echo "Usage: make <target>"
	@echo
	@echo " * 'docs' - produce document in the $(OUTPUT_DIRNAME) directory"
	@echo " * 'fmt' - format the json with indentation"
	@echo " * 'validate-examples' - validate the examples in the specification markdown files"
	@echo " * 'schema-fs' - regenerate the virtual schema http/FileSystem"
	@echo " * 'check-license' - check license headers in source files"
	@echo " * 'lint' - Execute the source code linter"
	@echo " * 'test' - Execute the unit tests"
	@echo " * 'img/*.png' - Generate PNG from dot file"

fmt:
	for i in schema/*.json ; do jq --indent 2 -M . "$${i}" > xx && cat xx > "$${i}" && rm xx ; done

docs: $(OUTPUT_DIRNAME)/$(DOC_FILENAME).pdf $(OUTPUT_DIRNAME)/$(DOC_FILENAME).html

ifeq "$(strip $(PANDOC))" ''
$(OUTPUT_DIRNAME)/$(DOC_FILENAME).pdf: $(DOC_FILES) $(FIGURE_FILES)
	$(error cannot build $@ without either pandoc or docker)
else
$(OUTPUT_DIRNAME)/$(DOC_FILENAME).pdf: $(DOC_FILES) $(FIGURE_FILES)
	@mkdir -p $(OUTPUT_DIRNAME)/ && \
	$(PANDOC) -f markdown_github -t latex --latex-engine=xelatex -o $(PANDOC_DST)$@ $(patsubst %,$(PANDOC_SRC)%,$(DOC_FILES))
	ls -sh $(shell readlink -f $@)

$(OUTPUT_DIRNAME)/$(DOC_FILENAME).html: header.html $(DOC_FILES) $(FIGURE_FILES)
	@mkdir -p $(OUTPUT_DIRNAME)/ && \
	cp -ap img/ $(shell pwd)/$(OUTPUT_DIRNAME)/&& \
	$(PANDOC) -f markdown_github -t html5 -H $(PANDOC_SRC)header.html --standalone -o $(PANDOC_DST)$@ $(patsubst %,$(PANDOC_SRC)%,$(DOC_FILES))
	ls -sh $(shell readlink -f $@)
endif

header.html: .tool/genheader.go specs-go/version.go
	go run .tool/genheader.go > $@

validate-examples: schema/fs.go
	go test -run TestValidate ./schema

schema/fs.go: $(wildcard schema/*.json) schema/gen.go
	cd schema && printf "%s\n\n%s\n" "$$(cat ../.header)" "$$(go generate)" > fs.go

schema-fs: schema/fs.go
	@echo "generating schema fs"

check-license:
	@echo "checking license headers"
	@./.tool/check-license

lint:
	@echo "checking lint"
	@./.tool/lint

test: schema/fs.go
	go test -race -cover $(shell go list ./... | grep -v /vendor/)

img/%.png: img/%.dot
	dot -Tpng $^ > $@


# When this is running in travis, it will only check the travis commit range
.gitvalidation:
	@which git-validation > /dev/null 2>/dev/null || (echo "ERROR: git-validation not found. Consider 'make install.tools' target" && false)
ifdef TRAVIS_COMMIT_RANGE
	git-validation -q -run DCO,short-subject,dangling-whitespace
else
	git-validation -v -run DCO,short-subject,dangling-whitespace -range $(EPOCH_TEST_COMMIT)..HEAD
endif

install.tools: $(TOOLS:%=.install.%)

.install.esc:
	go get -u github.com/mjibson/esc

.install.gitvalidation:
	go get -u github.com/vbatts/git-validation

.install.glide:
	go get -u github.com/Masterminds/glide

.install.glide-vc:
	go get -u github.com/sgotti/glide-vc

clean:
	rm -rf *~ $(OUTPUT_DIRNAME) header.html

.PHONY: \
	$(TOOLS:%=.install.%) \
	validate-examples \
	check-license \
	clean \
	lint \
	install.tools \
	docs \
	test \
	.gitvalidation \
	schema/fs.go \
	schema-fs
