
build: components index.js
	@component build --dev

components: component.json
	@component install --dev

clean:
	rm -fr build components

test:
	@./node_modules/.bin/mocha \
		--require should \
		--reporter spec

.PHONY: clean test
