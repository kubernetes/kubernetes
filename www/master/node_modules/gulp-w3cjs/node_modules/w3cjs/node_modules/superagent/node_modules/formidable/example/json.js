var common = require('../test/common'),
    http = require('http'),
    util = require('util'),
    formidable = common.formidable,
    Buffer = require('buffer').Buffer,
    port = common.port,
    server;

server = http.createServer(function(req, res) {
  if (req.method !== 'POST') {
    res.writeHead(200, {'content-type': 'text/plain'})
    res.end('Please POST a JSON payload to http://localhost:'+port+'/')
    return;
  }

  var form = new formidable.IncomingForm(),
      fields = {};

  form
    .on('error', function(err) {
      res.writeHead(500, {'content-type': 'text/plain'});
      res.end('error:\n\n'+util.inspect(err));
      console.error(err);
    })
    .on('field', function(field, value) {
      console.log(field, value);
      fields[field] = value;
    })
    .on('end', function() {
      console.log('-> post done');
      res.writeHead(200, {'content-type': 'text/plain'});
      res.end('received fields:\n\n '+util.inspect(fields));
    });
  form.parse(req);
});
server.listen(port);

console.log('listening on http://localhost:'+port+'/');


var request = http.request({
  host: 'localhost',
  path: '/',
  port: port,
  method: 'POST',
  headers: { 'content-type':'application/json', 'content-length':48 }
}, function(response) {
  var data = '';
  console.log('\nServer responded with:');
  console.log('Status:', response.statusCode);
  response.pipe(process.stdout);
  response.on('end', function() {
    console.log('\n')
    process.exit();
  });
  // response.on('data', function(chunk) {
  //   data += chunk.toString('utf8');
  // });
  // response.on('end', function() {
  //   console.log('Response Data:')
  //   console.log(data);
  //   process.exit();
  // });
})

request.write('{"numbers":[1,2,3,4,5],"nested":{"key":"value"}}');
request.end();
