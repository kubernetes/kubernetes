all: lint test coverage

clean:
	@rm -f coverage.html
	@rm -rf lib-cov

lint:
	@node_modules/.bin/jshint lib

test:
	@node_modules/.bin/mocha

coverage:
	@rm -rf lib-cov
	@jscoverage lib lib-cov
	@SAUCELABS_COV=1 node_modules/.bin/mocha --reporter html-cov > coverage.html

todo:
	@fgrep -H -e TODO -e FIXME -r lib test || true

.PHONY: clean lint test coverage todo
