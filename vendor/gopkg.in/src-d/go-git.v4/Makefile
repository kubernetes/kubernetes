# General
WORKDIR = $(PWD)

# Go parameters
GOCMD = go
GOTEST = $(GOCMD) test -v

# Git config
GIT_VERSION ?=
GIT_DIST_PATH ?= $(PWD)/.git-dist
GIT_REPOSITORY = http://github.com/git/git.git

# Coverage
COVERAGE_REPORT = coverage.txt
COVERAGE_PROFILE = profile.out
COVERAGE_MODE = atomic

ifneq ($(origin CI), undefined)
	WORKDIR := $(GOPATH)/src/gopkg.in/src-d/go-git.v4
endif

build-git:
	@if [ -f $(GIT_DIST_PATH)/git ]; then \
		echo "nothing to do, using cache $(GIT_DIST_PATH)"; \
	else \
		git clone $(GIT_REPOSITORY) -b $(GIT_VERSION) --depth 1 --single-branch $(GIT_DIST_PATH); \
		cd $(GIT_DIST_PATH); \
		make configure; \
		./configure; \
		make all; \
	fi

test:
	@cd $(WORKDIR); \
	$(GOTEST) ./...

test-coverage:
	@cd $(WORKDIR); \
	echo "" > $(COVERAGE_REPORT); \
	for dir in `find . -name "*.go" | grep -o '.*/' | sort | uniq`; do \
		$(GOTEST) $$dir -coverprofile=$(COVERAGE_PROFILE) -covermode=$(COVERAGE_MODE); \
		if [ $$? != 0 ]; then \
			exit 2; \
		fi; \
		if [ -f $(COVERAGE_PROFILE) ]; then \
			cat $(COVERAGE_PROFILE) >> $(COVERAGE_REPORT); \
			rm $(COVERAGE_PROFILE); \
		fi; \
	done; \

clean:
	rm -rf $(GIT_DIST_PATH)