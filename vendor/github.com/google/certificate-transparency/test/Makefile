test: apachectl mod_ssl.so
	$(MAKE) -C ..
	OPENSSLDIR=$(OPENSSLDIR) ./sslconnect_test.sh

apachectl:
	{ which apachectl && ln -s `which apachectl` $@; } || \
	{ which apache2ctl && ln -s `which apache2ctl` $@; } || \
	{ test -x /usr/sbin/apachectl && ln -s /usr/sbin/apachectl $@; } || \
	{ test -x /usr/sbin/apache2ctl && ln -s /usr/sbin/apache2ctl $@; } || \
	{ echo "could not find $@"; false; }

mod_ssl.so:
	find \
	  /usr/lib/apache2/modules \
	  /usr/local/libexec/apache22 \
	  -name 'mod_ssl.so' | { read x; \
	    { test -n "$$x" && ln -s "$$x" $@; } || \
	    { echo "could not find $@"; false; } }
