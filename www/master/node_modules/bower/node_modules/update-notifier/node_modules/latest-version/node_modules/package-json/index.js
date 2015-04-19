'use strict';
var got = require('got');
var registryUrl = require('registry-url');

module.exports = function (name, version, cb) {
	var url = registryUrl(name.split('/')[0]);

	if (typeof version !== 'string') {
		cb = version;
		version = '';
	}

	got(url + encodeURIComponent(name) + '/' + version, function (err, data) {
		if (err && err.code === 404) {
			cb(new Error('Package or version doesn\'t exist'));
			return;
		}

		if (err) {
			cb(err);
			return;
		}

		cb(null, JSON.parse(data));
	});
};
