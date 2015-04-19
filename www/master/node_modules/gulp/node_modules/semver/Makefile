files =  semver.browser.js \
         semver.min.js \
				 semver.browser.js.gz \
				 semver.min.js.gz

all: $(files)

clean:
	rm -f $(files)

semver.browser.js: head.js.txt semver.js foot.js.txt
	( cat head.js.txt; \
		cat semver.js | \
			egrep -v '^ *\/\* nomin \*\/' | \
			perl -pi -e 's/debug\([^\)]+\)//g'; \
		cat foot.js.txt ) > semver.browser.js

semver.min.js: semver.browser.js
	uglifyjs -m <semver.browser.js >semver.min.js

%.gz: %
	gzip --stdout -9 <$< >$@

.PHONY: all clean
