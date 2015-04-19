var http       = require('http'),
    formidable = require('../../lib/index'),
    request    = require('request'),
    assert     = require('assert');

var host = 'localhost';

var index = [
  '<form action="/" method="post" enctype="multipart/form-data">',
  '  <input type="text" name="foo" />',
  '  <input type="submit" />',
  '</form>'
].join("\n");

var server = http.createServer(function(req, res) {

  // Show a form for testing purposes.
  if (req.method == 'GET') {
    res.writeHead(200, {'content-type': 'text/html'});
    res.end(index);
    return;
  }

  // Parse form and write results to response.
  var form = new formidable.IncomingForm();
  form.parse(req, function(err, fields, files) {
    res.writeHead(200, {'content-type': 'text/plain'}); 
    res.write(JSON.stringify({err: err, fields: fields, files: files}));
    res.end();
  });

}).listen(0, host, function() {

  console.log("Server up and running...");

  var server = this,
      url    = 'http://' + host + ':' + server.address().port;

  var parts  = [
    {'Content-Disposition': 'form-data; name="foo"', 'body': 'bar'}
  ]

  var req = request({method: 'POST', url: url, multipart: parts}, function(e, res, body) {
    var obj = JSON.parse(body);
    assert.equal("bar", obj.fields.foo);
    server.close();
  });

});
