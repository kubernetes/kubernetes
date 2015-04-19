ALL_TESTS = $(shell find test/ -name '*.test.js')
ALL_INTEGRATION = $(shell find test/ -name '*.integration.js')

all:
	node-gyp configure build

clean:
	node-gyp clean

run-tests:
	@./node_modules/.bin/mocha \
		-t 2000 \
		-s 2400 \
		$(TESTFLAGS) \
		$(TESTS)

run-integrationtests:
	@./node_modules/.bin/mocha \
		-t 5000 \
		-s 6000 \
		$(TESTFLAGS) \
		$(TESTS)

test:
	@$(MAKE) NODE_TLS_REJECT_UNAUTHORIZED=0 NODE_PATH=lib TESTS="$(ALL_TESTS)" run-tests

integrationtest:
	@$(MAKE) NODE_TLS_REJECT_UNAUTHORIZED=0 NODE_PATH=lib TESTS="$(ALL_INTEGRATION)" run-integrationtests

benchmark:
	@node bench/sender.benchmark.js
	@node bench/parser.benchmark.js

autobahn:
	@NODE_PATH=lib node test/autobahn.js

autobahn-server:
	@NODE_PATH=lib node test/autobahn-server.js

.PHONY: test
