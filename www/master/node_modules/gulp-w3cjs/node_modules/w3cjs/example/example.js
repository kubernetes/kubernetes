var w3cjs = require('w3cjs');

var results = w3cjs.validate({
	file: 'demo.html', // file can either be a local file or a remote file
	//file: 'http://html5boilerplate.com/',
//	input: '<html>...</html>',
//	input: myBuffer,
	output: 'json', // Defaults to 'json', other option includes html
   doctype: 'HTML5', // Defaults false for autodetect
   charset: 'utf-8', // Defaults false for autodetect
   proxy: 'http://proxy:8080', // Default to null
	callback: function (res) {
		console.log(res);
		// depending on the output type, res will either be a json object or a html string
	}
});