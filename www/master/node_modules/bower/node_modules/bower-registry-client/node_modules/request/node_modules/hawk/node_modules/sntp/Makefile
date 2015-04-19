test:
	@./node_modules/.bin/lab
test-cov: 
	@./node_modules/.bin/lab -r threshold -t 100
test-cov-html:
	@./node_modules/.bin/lab -r html -o coverage.html
complexity:
	@./node_modules/.bin/cr -o complexity.md -f markdown lib

.PHONY: test test-cov test-cov-html complexity

