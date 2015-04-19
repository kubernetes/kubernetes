'use strict';
var qs = require('querystring');

/**
 * Tracking providers.
 *
 * Each provider is a function(id, path) that should return
 * options object for request() call. It will be called bound
 * to Insight instance object.
 */

module.exports = {
	// Google Analytics — https://www.google.com/analytics/
	google: function (id, path) {
		var now = Date.now();

		var _qs = {
			v: 1, // GA Measurement Protocol API version
			t: 'pageview', // hit type
			aip: 1, // anonymize IP
			tid: this.trackingCode,
			cid: this.clientId, // random UUID
			cd1: this.os,
			// GA custom dimension 2 = Node Version, scope = Session
			cd2: this.nodeVersion,
			// GA custom dimension 3 = App Version, scope = Session (temp solution until refactored to work w/ GA app tracking)
			cd3: this.appVersion,
			dp: path,
			qt: now - parseInt(id, 10), // queue time - delta (ms) between now and track time
			z: now // cache busting, need to be last param sent
		};

		return {
			url: 'https://ssl.google-analytics.com/collect',
			method: 'POST',
			// GA docs recommends body payload via POST instead of querystring via GET
			body: qs.stringify(_qs)
		};
	},
	// Yandex.Metrica — http://metrica.yandex.com
	yandex: function (id, path) {
		var request = require('request');

		var ts = new Date(parseInt(id, 10))
			.toISOString()
			.replace(/[-:T]/g, '')
			.replace(/\..*$/, '');

		var qs = {
			wmode: 3,
			ut: 'noindex',
			'page-url': 'http://' + this.packageName + '.insight' + path + '?version=' + this.packageVersion,
			'browser-info': 'i:' + ts + ':z:0:t:' + path,
			rn: Date.now() // cache busting
		};

		var url = 'https://mc.yandex.ru/watch/' + this.trackingCode;

		// set custom cookie using tough-cookie
		var _jar = request.jar();
		var cookieString = 'name=yandexuid; value=' + this.clientId + '; path=/;';
		var cookie = request.cookie(cookieString);
		_jar.setCookie(cookie, url);

		return {
			url: url,
			method: 'GET',
			qs: qs,
			jar: _jar
		};
	}
};
