VENDOR_BUNDLE = vendor/bundle

build: $(VENDOR_BUNDLE)/.bundled
	BEEFCAKE_NAMESPACE=Prometheus::Client protoc --beefcake_out lib/prometheus/client/model -I .. ../metrics.proto

$(VENDOR_BUNDLE):
	mkdir -p $@

$(VENDOR_BUNDLE)/.bundled: $(VENDOR_BUNDLE) Gemfile
	bundle install --quiet --path $<
	@touch $@

clean:
	-rm -f lib/prometheus/client/model/metrics.pb.rb
	-rm -rf $(VENDOR_BUNDLE)

.PHONY: build clean
