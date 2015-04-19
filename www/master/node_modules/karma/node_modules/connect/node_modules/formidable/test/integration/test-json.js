var common = require('../common');
var formidable = common.formidable;
var http = require('http');
var assert = require('assert');

var testData = {
  numbers: [1, 2, 3, 4, 5],
  nested: { key: 'value' }
};

var server = http.createServer(function(req, res) {
    var form = new formidable.IncomingForm();

    form.parse(req, function(err, fields, files) {
        assert.deepEqual(fields, testData);

        res.end();
        server.close();
    });
});

var port = common.port;

server.listen(port, function(err){
    assert.equal(err, null);

    var request = http.request({
        port: port,
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    });

    request.write(JSON.stringify(testData));
    request.end();
});

