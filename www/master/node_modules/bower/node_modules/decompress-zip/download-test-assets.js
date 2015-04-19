'use strict';
var fs = require('fs');
var request = require('request');
var tmp = require('tmp');
var exec = require('child_process').exec;
var path = require('path');

var url = 'https://drive.google.com/uc?id=0Bxxp2pVhWG1DTFNWQ1hsSkZKZmM&export=download';

var errorHandler = function (err) {
    throw err;
};

var extract = function (filename) {
    exec('tar -xvzf ' + filename, {
        cwd: path.join(__dirname, 'test'),
        maxBuffer: 1024 * 1024
    }, function (err, stdout, stderr) {
        if (err) {
            throw err;
        }

        console.log('Done');
    });
};

tmp.file({
    prefix: 'assets',
    postfix: '.tgz'
}, function (err, filename, fd) {
    console.log('Downloading ' + url + ' to ' + filename);

    var read = request(url);
    var write = fs.createWriteStream(filename);

    read.on('error', errorHandler);
    write.on('error', errorHandler);

    // For node 0.8 we can't just use the 'finish' event of the pipe
    read.on('end', function () {
        write.end(extract.bind(null, filename));
    });

    read.pipe(write, {end: false});
});
