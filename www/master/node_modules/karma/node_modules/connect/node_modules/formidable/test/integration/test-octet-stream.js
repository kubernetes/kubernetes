var common = require('../common');
var formidable = common.formidable;
var http = require('http');
var fs = require('fs');
var path = require('path');
var hashish = require('hashish');
var assert = require('assert');

var testFilePath = path.join(__dirname, '../fixture/file/binaryfile.tar.gz');

var server = http.createServer(function(req, res) {
    var form = new formidable.IncomingForm();

    form.parse(req, function(err, fields, files) {
        assert.equal(hashish(files).length, 1);
        var file = files.file;

        assert.equal(file.size, 301);

        var uploaded = fs.readFileSync(file.path);
        var original = fs.readFileSync(testFilePath);

        assert.deepEqual(uploaded, original);

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
            'Content-Type': 'application/octet-stream'
        }
    });

    fs.createReadStream(testFilePath).pipe(request);
});

