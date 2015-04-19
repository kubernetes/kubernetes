var request = require('superagent');
var proxyRequest = require('superagent-proxy')(request);

var fs = require('fs');

var w3cCheckUrl = 'http://validator.w3.org/check';
var defaultOutput = 'json';
var defaultDoctype = null;
var defaultCharset = null;
var defaultProxy = null;

var defaultCallback = function (res) {
	console.log(res);
}

function setW3cCheckUrl(newW3cCheckUrl) {
	w3cCheckUrl = newW3cCheckUrl;
}

function validate(options) {
	var output = options.output || defaultOutput;
	var callback = options.callback || defaultCallback;
	var doctype = options.doctype || defaultDoctype;
	var charset = options.charset || defaultCharset;
	var file = options.file;
	var input = options.input;
	var context = '';

	var type;
	if(typeof input !== 'undefined'){
		type = 'string';
		context = input;
	} else if(typeof file !== 'undefined' && (file.substr(0,5) === 'http:' || file.substr(0, 6) === 'https:')){
		type = 'remote';
		context = file;
	} else if(typeof file !== 'undefined'){
		type = 'local';
		context = file;
	} else {
		return false;
	}

	var req = getRequest(type !== 'remote', options);

	if(type === 'remote') {
		req.query({ output: output });
		req.query({ uri: file });
		if(doctype !== null) req.query({doctype: doctype});
		if(charset !== null) req.query({charset: charset});
	} else {
		req.field('output', output);
		req.field('uploaded_file', (type === 'local') ? fs.readFileSync(file, 'utf8') : input);
		if(doctype !== null) req.field('doctype', doctype);
		if(charset !== null) req.field('charset', charset);
	};
	req.end(function(res){
		if(output === 'json'){
			res.body.context = context;
			callback(res.body);
		} else {
			callback(res.text);
		}
	});
}

var getRequest = function(isLocal, options) {
   var req = isLocal ? proxyRequest.post(w3cCheckUrl) : proxyRequest.get(w3cCheckUrl);

   var proxy = options.proxy || defaultProxy;
   if (proxy !== null) {
      req.proxy(proxy);
   }

   req.set('User-Agent', 'w3cjs - npm module');

   return req;
}

var w3cjs = {
	validate: validate,
	setW3cCheckUrl: setW3cCheckUrl
}
if (typeof exports !== 'undefined') {
  if (typeof module !== 'undefined' && module.exports) {
	exports = module.exports = w3cjs;
  }
  exports.w3cjs = w3cjs
} else {
  root['w3cjs'] = w3cjs;
}
